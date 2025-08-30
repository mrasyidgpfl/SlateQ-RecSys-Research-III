# src/agents/slateq_dueling_noisynet_agent.py
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy


# ---------- Noisy layer (Factorised Gaussian) ----------

class NoisyDense(tf.keras.layers.Layer):
    """Factorised Gaussian NoisyNet layer per Fortunato et al. 2017."""
    def __init__(self, units, sigma0=0.5, activation=None, use_bias=True, kernel_regularizer=None, name=None):
        super().__init__(name=name)
        self.units = int(units)
        self.sigma0 = float(sigma0)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = bool(use_bias)
        self.kernel_regularizer = kernel_regularizer
        self._in_features = None

    def build(self, input_shape):
        in_feats = int(input_shape[-1])
        self._in_features = in_feats

        # Init ranges as in standard practice for NoisyNet
        mu_range = 1.0 / tf.math.sqrt(tf.cast(in_feats, tf.float32))

        self.w_mu = self.add_weight(
            name="w_mu", shape=(in_feats, self.units),
            initializer=tf.keras.initializers.RandomUniform(minval=-mu_range, maxval=mu_range),
            regularizer=self.kernel_regularizer, trainable=True
        )
        self.w_sigma = self.add_weight(
            name="w_sigma", shape=(in_feats, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma0 / tf.sqrt(tf.cast(in_feats, tf.float32))),
            trainable=True
        )

        if self.use_bias:
            self.b_mu = self.add_weight(
                name="b_mu", shape=(self.units,),
                initializer=tf.keras.initializers.RandomUniform(minval=-mu_range, maxval=mu_range),
                regularizer=self.kernel_regularizer, trainable=True
            )
            self.b_sigma = self.add_weight(
                name="b_sigma", shape=(self.units,),
                initializer=tf.keras.initializers.Constant(self.sigma0 / tf.sqrt(tf.cast(self.units, tf.float32))),
                trainable=True
            )

        super().build(input_shape)

    @staticmethod
    def _f(eps):
        # factorised noise transform: f(x) = sign(x) * sqrt(|x|)
        return tf.sign(eps) * tf.sqrt(tf.abs(eps) + 1e-8)

    def _sample_noise(self, batch_size=None):
        eps_in = tf.random.normal([self._in_features])
        eps_out = tf.random.normal([self.units])
        f_in = NoisyDense._f(eps_in)    # [in]
        f_out = NoisyDense._f(eps_out)  # [out]
        # outer product for factorised noise
        w_noise = tf.einsum("i,j->ij", f_in, f_out)  # [in, out]
        if self.use_bias:
            b_noise = f_out  # [out]
        else:
            b_noise = None
        return w_noise, b_noise

    def call(self, x, training=False):
        if training:
            w_noise, b_noise = self._sample_noise()
            w = self.w_mu + self.w_sigma * w_noise
            y = tf.linalg.matmul(x, w)
            if self.use_bias:
                y = y + (self.b_mu + self.b_sigma * b_noise)
        else:
            y = tf.linalg.matmul(x, self.w_mu)
            if self.use_bias:
                y = y + self.b_mu

        if self.activation is not None:
            y = self.activation(y)
        return y


# ---------- Dueling Noisy SlateQ network ----------

class DuelingNoisySlateQNetwork(network.Network):
    """interest [B, T] -> per-item Q-values [B, N] with Noisy trunk and dueling head."""
    def __init__(self, input_tensor_spec, num_items: int,
                 hidden=(256, 128, 64), sigma0=0.5, l2=0.0, name="DuelingNoisySlateQNetwork"):
        super().__init__(input_tensor_spec={"interest": input_tensor_spec}, state_spec=(), name=name)
        self._num_items = int(num_items)
        reg = tf.keras.regularizers.l2(l2) if (l2 and l2 > 0) else None

        layers = []
        for h in hidden:
            layers.append(NoisyDense(h, sigma0=sigma0, activation="relu", kernel_regularizer=reg))
        self._trunk = tf.keras.Sequential(layers, name=f"{name}_trunk")

        # Dueling heads are also noisy
        self._value = NoisyDense(1, sigma0=sigma0, activation=None, kernel_regularizer=reg, name=f"{name}_value")
        self._advantage = NoisyDense(self._num_items, sigma0=sigma0, activation=None, kernel_regularizer=reg, name=f"{name}_advantage")

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = tf.convert_to_tensor(inputs["interest"])
        z = self._trunk(x, training=training)
        v = self._value(z, training=training)
        a = self._advantage(z, training=training)
        a_centered = a - tf.reduce_mean(a, axis=1, keepdims=True)
        q_values = v + a_centered
        return q_values, network_state


# ---------- Policies that can toggle Noisy forward during action ----------

class NoisyAwareSlateQPolicy(tf_policy.TFPolicy):
    """Greedy top-k slate by ranking v(s,i) * Q(s,i). Can forward the network with training=True to enable NoisyNet."""
    def __init__(self, time_step_spec, action_spec, q_network, slate_size: int, beta: float = 5.0, use_noisy: bool = True):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = int(slate_size)
        self._beta = float(beta)
        self._use_noisy = bool(use_noisy)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation
        # training flag toggles NoisyNet sampling
        q_values, _ = self._q_network(obs, time_step.step_type, training=self._use_noisy)

        interest   = tf.convert_to_tensor(obs["interest"])
        item_feats = tf.convert_to_tensor(obs["item_features"])
        item_feats = tf.cond(
            tf.equal(tf.rank(item_feats), 2),
            lambda: tf.tile(tf.expand_dims(item_feats, 0), [tf.shape(interest)[0], 1, 1]),
            lambda: item_feats
        )

        v_all = tf.einsum("bt,bnt->bn", interest, item_feats)  # [B, N]
        score = v_all * q_values
        top_k = tf.math.top_k(score, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=top_k.indices)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


# ---------- Agent ----------

class SlateQDuelingNoisyNetAgent(tf_agent.TFAgent):
    """
    SlateQ + Dueling + NoisyNet.
    - Noisy layers drive exploration, so no epsilon is required.
    - Learns on no-clicks via expected-on-slate Q at time t.
    - Double Q with expected next-slate value.
    - Huber loss, grad clipping, reward squashing with tanh.
    """
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=1e-3,
                 target_update_period=1000, tau=0.005, gamma=0.95, beta=5.0,
                 huber_delta=1.0, grad_clip_norm=10.0, reward_scale=10.0,
                 l2=0.0, pos_weights=None,
                 noisy_sigma0=0.5, noisy_eval_collect=True, noisy_eval_eval=False):
        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self._target_update_period = int(target_update_period)
        self._tau = float(tau)
        self._gamma = tf.constant(gamma, dtype=tf.float32)
        self._beta = tf.constant(beta, dtype=tf.float32)
        self._grad_clip_norm = float(grad_clip_norm)
        self._reward_scale = float(reward_scale)  # kept for compat; we squash with tanh
        self.is_learning = True

        # position weights for choice within slate
        if pos_weights is None:
            pw = tf.linspace(1.0, 0.3, self._slate_size)
        else:
            pw = tf.convert_to_tensor(pos_weights, dtype=tf.float32)
            cur = tf.shape(pw)[0]
            def pad():
                pad_len = self._slate_size - cur
                last = pw[-1]
                return tf.concat([pw, tf.fill([pad_len], last)], axis=0)
            def trunc():
                return pw[:self._slate_size]
            pw = tf.cond(cur < self._slate_size, pad, trunc)
        self._pos_w = tf.reshape(pw, [1, self._slate_size])

        input_tensor_spec = time_step_spec.observation["interest"]

        # Noisy Q networks
        self._q_network = DuelingNoisySlateQNetwork(input_tensor_spec, num_items, sigma0=noisy_sigma0, l2=l2)
        self._target_q_network = DuelingNoisySlateQNetwork(input_tensor_spec, num_items, sigma0=noisy_sigma0, l2=l2)

        # build once
        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network.set_weights(self._q_network.get_weights())

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)

        # Policies: collect uses noise, eval optionally uses noise
        self._policy = NoisyAwareSlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size,
            beta=float(self._beta.numpy()) if hasattr(self._beta, "numpy") else self._beta,
            use_noisy=bool(noisy_eval_eval),
        )
        self._collect_policy = NoisyAwareSlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size,
            beta=float(self._beta.numpy()) if hasattr(self._beta, "numpy") else self._beta,
            use_noisy=bool(noisy_eval_collect),
        )

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=self._policy,
            collect_policy=self._collect_policy,
            train_sequence_length=2,
            train_step_counter=self._train_step_counter
        )

    @property
    def collect_data_spec(self):
        # full observation spec; main.py will store everything unless you add a caching path
        return trajectory_lib.Trajectory(
            step_type=self._time_step_spec.step_type,
            observation=self._time_step_spec.observation,
            action=self._action_spec,
            policy_info=(),
            next_step_type=self._time_step_spec.step_type,
            reward=self._time_step_spec.reward,
            discount=self._time_step_spec.discount,
        )

    def _initialize(self):
        return tf.no_op()

    # ---------- helpers ----------
    @staticmethod
    def _slice_time(x, t_idx: int):
        return x[:, t_idx]

    @staticmethod
    def _flatten_keep_tail(x, tail_ndims: int):
        x = tf.convert_to_tensor(x)
        shape = tf.shape(x)
        rank = tf.rank(x)
        def flatten_all():
            return tf.reshape(x, [-1])
        def flatten_keep():
            if tail_ndims == 0:
                return tf.reshape(x, [-1])
            lead = tf.reduce_prod(shape[:-tail_ndims])
            tail = shape[-tail_ndims:]
            return tf.reshape(x, tf.concat([[lead], tail], axis=0))
        return tf.cond(tf.less(rank, tail_ndims), flatten_all, flatten_keep)

    @staticmethod
    def _align_leading(x, B):
        x = tf.convert_to_tensor(x)
        xB = tf.shape(x)[0]
        def same():
            return x
        def repeat_or_trunc():
            idx = tf.math.mod(tf.range(B), xB)
            return tf.gather(x, idx, axis=0)
        return tf.cond(tf.equal(xB, B), same, repeat_or_trunc)

    @staticmethod
    def _to_BNT(feats_any, B, N, T):
        rnk = tf.rank(feats_any)
        def from4():
            x = SlateQDuelingNoisyNetAgent._flatten_keep_tail(feats_any, tail_ndims=2)
            return SlateQDuelingNoisyNetAgent._align_leading(x, B)
        def from3():
            x = SlateQDuelingNoisyNetAgent._flatten_keep_tail(feats_any, tail_ndims=2)
            return SlateQDuelingNoisyNetAgent._align_leading(x, B)
        def from2():
            x = tf.expand_dims(feats_any, 0)
            return tf.tile(x, [B, 1, 1])
        x = tf.case(
            [(tf.equal(rnk, 4), from4),
             (tf.equal(rnk, 3), from3)],
            default=from2
        )
        return tf.reshape(x, [B, N, T])

    # ---------- training ----------
    def _train(self, experience, weights=None):
        s = SlateQDuelingNoisyNetAgent
        slice_time = s._slice_time
        flatten_keep_tail = s._flatten_keep_tail
        align_leading = s._align_leading
        to_BNT = s._to_BNT

        # unpack
        interest_t_flat   = flatten_keep_tail(slice_time(experience.observation["interest"], 0), tail_ndims=1)
        interest_tp1_flat = flatten_keep_tail(slice_time(experience.observation["interest"], 1), tail_ndims=1)
        B = tf.shape(interest_tp1_flat)[0]
        T = tf.shape(interest_tp1_flat)[1]
        N = tf.constant(self._num_items, dtype=tf.int32)

        obs_t   = {"interest": align_leading(interest_t_flat, B)}
        obs_tp1 = {"interest": align_leading(interest_tp1_flat, B)}

        act_t_any = slice_time(experience.action, 0)
        action_flat = flatten_keep_tail(act_t_any, tail_ndims=1)
        action_t = align_leading(action_flat, B)  # [B, K]

        click_pos_any = slice_time(experience.observation["choice"], 1)
        click_pos_flat = flatten_keep_tail(click_pos_any, tail_ndims=0)
        click_pos_t = tf.cast(align_leading(click_pos_flat, B), tf.int32)

        item_feats_t_any   = slice_time(experience.observation["item_features"], 0)
        item_feats_tp1_any = slice_time(experience.observation["item_features"], 1)
        item_feats_t   = to_BNT(item_feats_t_any,   B, N, T)
        item_feats_tp1 = to_BNT(item_feats_tp1_any, B, N, T)

        batch_size = tf.shape(action_t)[0]
        slate_size = tf.shape(action_t)[1]

        clicked_mask = tf.less(click_pos_t, slate_size)  # [B]
        safe_click_pos = tf.minimum(click_pos_t, slate_size - 1)
        row_idx = tf.range(batch_size, dtype=tf.int32)
        idx = tf.stack([row_idx, safe_click_pos], axis=1)
        clicked_item_id = tf.gather_nd(action_t, idx)  # [B]

        with tf.GradientTape() as tape:
            # Q at time t with NoisyNet active during training
            q_tm1_all, _ = self._q_network(obs_t, training=True)  # [B, N]
            q_clicked = tf.gather(q_tm1_all, clicked_item_id, axis=1, batch_dims=1)  # [B]

            # Expected-on-slate at time t for no-click learning
            q_on_slate_t = tf.gather(q_tm1_all, action_t, axis=1, batch_dims=1)      # [B, K]
            v_all_t = tf.einsum("bt,bnt->bn", obs_t["interest"], item_feats_t)       # [B, N]
            v_on_slate_t = tf.gather(v_all_t, action_t, axis=1, batch_dims=1)        # [B, K]
            p_t = tf.nn.softmax(self._beta * v_on_slate_t, axis=-1)                  # [B, K]
            p_t = p_t * self._pos_w
            p_t = p_t / (tf.reduce_sum(p_t, axis=-1, keepdims=True) + 1e-8)
            q_expected_t = tf.reduce_sum(p_t * q_on_slate_t, axis=-1)                # [B]

            # Double Q target on next state using expected next-slate value
            q_tp1_online_all, _ = self._q_network(obs_tp1, training=True)            # [B, N]
            v_all_tp1 = tf.einsum("bt,bnt->bn", obs_tp1["interest"], item_feats_tp1) # [B, N]
            score_next = v_all_tp1 * q_tp1_online_all                                 # [B, N]
            next_topk_idx = tf.math.top_k(score_next, k=self._slate_size).indices     # [B, K]

            q_tp1_target_all, _ = self._target_q_network(obs_tp1, training=False)     # [B, N]
            q_next_on_slate = tf.gather(q_tp1_target_all, next_topk_idx, axis=1, batch_dims=1)  # [B, K]

            v_next_on_slate = tf.gather(v_all_tp1, next_topk_idx, axis=1, batch_dims=1)        # [B, K]
            p_next = tf.nn.softmax(self._beta * v_next_on_slate, axis=-1)                       # [B, K]
            p_next = p_next * self._pos_w
            p_next = p_next / (tf.reduce_sum(p_next, axis=-1, keepdims=True) + 1e-8)

            expect_next = tf.reduce_sum(p_next * q_next_on_slate, axis=-1)  # [B]

            # rewards and discount
            reward_t_any = slice_time(experience.reward, 1)
            discount_tp1_any = tf.cast(slice_time(experience.discount, 1), tf.float32)
            reward_flat = flatten_keep_tail(reward_t_any, tail_ndims=0)
            discount_flat = flatten_keep_tail(discount_tp1_any, tail_ndims=0)
            r_raw = align_leading(reward_flat, B)
            r_scaled = tf.tanh(r_raw)  # squash to keep targets healthy
            discount_tp1 = align_leading(discount_flat, B)

            y = tf.stop_gradient(r_scaled + self._gamma * discount_tp1 * expect_next)  # [B]

            # clicked Q if clicked, else expected-on-slate at time t
            q_pred_t = tf.where(clicked_mask, q_clicked, q_expected_t)  # [B]

            per_example = self._huber(y, q_pred_t)
            loss = tf.reduce_mean(per_example)

        grads = tape.gradient(loss, self._q_network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self._grad_clip_norm)
        self._optimizer.apply_gradients(zip(grads, self._q_network.trainable_variables))
        self._train_step_counter.assign_add(1)

        # Target updates
        if self._tau > 0.0:
            for w_t, w in zip(self._target_q_network.weights, self._q_network.weights):
                w_t.assign(self._tau * w + (1.0 - self._tau) * w_t)
        else:
            step = int(self._train_step_counter.numpy())
            if step % self._target_update_period == 0:
                self._target_q_network.set_weights(self._q_network.get_weights())

        return tf_agent.LossInfo(loss=loss, extra={})
