import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy


# NoisyNet building blocks
class NoisyDense(tf.keras.layers.Layer):
    """
    Factorized Gaussian NoisyNet layer (Fortunato et al., 2018).
    """
    def __init__(self, units, activation=None, sigma0=0.5, use_bias=True, name=None):
        super().__init__(name=name)
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.sigma0 = float(sigma0)
        self.use_bias = bool(use_bias)
        self._eps_in = None
        self._eps_out = None

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        mu_range = 1.0 / tf.math.sqrt(tf.cast(in_dim, tf.float32))
        self.w_mu = self.add_weight(
            'w_mu', shape=(in_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(minval=-mu_range, maxval=mu_range),
            trainable=True
        )
        self.w_sigma = self.add_weight(
            'w_sigma', shape=(in_dim, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma0 / tf.math.sqrt(tf.cast(in_dim, tf.float32))),
            trainable=True
        )
        if self.use_bias:
            self.b_mu = self.add_weight(
                'b_mu', shape=(self.units,),
                initializer=tf.keras.initializers.RandomUniform(minval=-mu_range, maxval=mu_range),
                trainable=True
            )
            self.b_sigma = self.add_weight(
                'b_sigma', shape=(self.units,),
                initializer=tf.keras.initializers.Constant(self.sigma0 / tf.math.sqrt(tf.cast(in_dim, tf.float32))),
                trainable=True
            )

    @staticmethod
    def _f(x):
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def reset_noise(self, batch_size=None):
        eps_in = tf.random.normal((self.w_mu.shape[0],), dtype=tf.float32)
        eps_out = tf.random.normal((self.w_mu.shape[1],), dtype=tf.float32)
        self._eps_in = self._f(eps_in)
        self._eps_out = self._f(eps_out)

    def call(self, inputs, training=False):
        if training:
            if self._eps_in is None or self._eps_out is None:
                self.reset_noise()
            noise_w = tf.einsum('i,j->ij', self._eps_in, self._eps_out)
            w = self.w_mu + self.w_sigma * noise_w
            if self.use_bias:
                b = self.b_mu + self.b_sigma * self._eps_out
            else:
                b = None
        else:
            w = self.w_mu
            b = self.b_mu if self.use_bias else None

        y = tf.linalg.matmul(inputs, w)
        if b is not None:
            y = y + b
        if self.activation is not None:
            y = self.activation(y)
        return y


# Per-item Q network (Noisy)
class NoisySlateQNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 num_items: int,
                 hidden=(256, 128, 64),
                 sigma0=0.5,
                 name='NoisySlateQNetwork'):
        super().__init__(input_tensor_spec={'interest': input_tensor_spec},
                         state_spec=(),
                         name=name)
        self._num_items = int(num_items)
        self._sigma0 = float(sigma0)

        self._layers = []
        for i, h in enumerate(hidden):
            self._layers.append(
                NoisyDense(h, activation='relu', sigma0=self._sigma0, name=f"{name}_noisy_{i}")
            )
        self._head = NoisyDense(self._num_items, activation=None, sigma0=self._sigma0, name=f"{name}_head")

    def reset_noise(self):
        for lyr in self._layers:
            lyr.reset_noise()
        self._head.reset_noise()

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = tf.convert_to_tensor(inputs['interest'])
        for lyr in self._layers:
            x = lyr(x, training=training)
        q_values = self._head(x, training=training)
        return q_values, network_state


# Deterministic top-K policy (exploration via noisy weights)
class NoisySlateQPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, q_network, slate_size: int, beta: float = 5.0, noisy_eval: bool = True):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = int(slate_size)
        self._beta = float(beta)
        self._noisy_eval = bool(noisy_eval)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation
        if hasattr(self._q_network, "reset_noise"):
            self._q_network.reset_noise()

        q_values, _ = self._q_network(obs, time_step.step_type, training=self._noisy_eval)

        interest = tf.convert_to_tensor(obs["interest"])
        item_feats = tf.convert_to_tensor(obs["item_features"])

        item_feats = tf.cond(
            tf.equal(tf.rank(item_feats), 2),
            lambda: tf.tile(tf.expand_dims(item_feats, 0), [tf.shape(interest)[0], 1, 1]),
            lambda: item_feats
        )

        v_all = tf.einsum("bt,bnt->bn", interest, item_feats)
        score = v_all * q_values
        top_k = tf.math.top_k(score, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=top_k.indices)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


# SlateQ with NoisyNet Agent
class SlateQNoisyNetAgent(tf_agent.TFAgent):
    """
    SlateQ with NoisyNet exploration:
      - No epsilon. Exploration arises from learned weight noise.
      - Double-Q target: argmax via online net. Target evaluated via target net.
      - Next-slate expectation uses v(s,i) and position weights (cascade alignment).
      - Soft target updates (tau) or hard copies by period.
      - Huber loss, gradient clipping, reward scaling.
    """
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=1e-3,
                 target_update_period=1000,
                 tau=0.005,
                 gamma=0.95,
                 beta=5.0,
                 huber_delta=1.0,
                 grad_clip_norm=10.0,
                 reward_scale=10.0,
                 noisy_sigma0=0.5,
                 pos_weights=None,
                 noisy_eval_collect=True,
                 noisy_eval_eval=True):
        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self._target_update_period = int(target_update_period)
        self._tau = float(tau)
        self._gamma = tf.constant(gamma, dtype=tf.float32)
        self._beta = tf.constant(beta, dtype=tf.float32)
        self._grad_clip_norm = float(grad_clip_norm)
        self._reward_scale = float(reward_scale)
        self.is_learning = True

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

        input_tensor_spec = time_step_spec.observation['interest']

        self._q_network = NoisySlateQNetwork(input_tensor_spec, num_items, sigma0=noisy_sigma0)
        self._target_q_network = NoisySlateQNetwork(input_tensor_spec, num_items, sigma0=noisy_sigma0)

        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network.set_weights(self._q_network.get_weights())

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._huber = tf.keras.losses.Huber(delta=huber_delta,
                                            reduction=tf.keras.losses.Reduction.NONE)

        self._policy = NoisySlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size,
            beta=float(self._beta.numpy()) if hasattr(self._beta, "numpy") else self._beta,
            noisy_eval=noisy_eval_eval
        )
        self._collect_policy = NoisySlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size,
            beta=float(self._beta.numpy()) if hasattr(self._beta, "numpy") else self._beta,
            noisy_eval=noisy_eval_collect
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

    def _train(self, experience, weights=None):
        def slice_time(x, t_idx: int):
            return x[:, t_idx]

        def flatten_keep_tail(x, tail_ndims: int):
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

        obs_t_interest    = slice_time(experience.observation['interest'], 0)
        obs_tp1_interest  = slice_time(experience.observation['interest'], 1)
        click_pos_t       = slice_time(experience.observation['choice'], 1)

        item_feats_tp1_any = slice_time(experience.observation['item_features'], 1)
        item_feats_rank = tf.rank(item_feats_tp1_any)
        item_feats_tp1 = tf.case([
            (tf.equal(item_feats_rank, 4), lambda: item_feats_tp1_any[0, 0]),
            (tf.equal(item_feats_rank, 3), lambda: item_feats_tp1_any[0]),
        ], default=lambda: item_feats_tp1_any)

        act = experience.action
        act_rank = tf.rank(act)
        action_t = tf.cond(
            tf.greater_equal(act_rank, 2),
            lambda: act[:, 0],
            lambda: act
        )

        reward_t      = slice_time(experience.reward, 1)
        discount_tp1  = tf.cast(slice_time(experience.discount, 1), tf.float32)

        obs_t   = {'interest': flatten_keep_tail(obs_t_interest,   tail_ndims=1)}
        obs_tp1 = {'interest': flatten_keep_tail(obs_tp1_interest, tail_ndims=1)}
        action_t     = flatten_keep_tail(action_t,     tail_ndims=1)
        click_pos_t  = tf.cast(flatten_keep_tail(click_pos_t,  tail_ndims=0), tf.int32)
        reward_t     = flatten_keep_tail(reward_t,     tail_ndims=0)
        discount_tp1 = flatten_keep_tail(discount_tp1, tail_ndims=0)

        batch_size = tf.shape(action_t)[0]
        slate_size = tf.shape(action_t)[1]

        clicked_mask = tf.less(click_pos_t, slate_size)
        safe_click_pos = tf.minimum(click_pos_t, slate_size - 1)
        idx = tf.stack([tf.range(batch_size), safe_click_pos], axis=1)
        clicked_item_id = tf.gather_nd(action_t, idx)

        with tf.GradientTape() as tape:
            if hasattr(self._q_network, "reset_noise"):
                self._q_network.reset_noise()

            q_tm1_all, _ = self._q_network(obs_t, training=True)
            q_clicked = tf.gather(q_tm1_all, clicked_item_id, axis=1, batch_dims=1)

            if hasattr(self._q_network, "reset_noise"):
                self._q_network.reset_noise()
            q_tp1_online_all, _ = self._q_network(obs_tp1, training=True)

            v_all = tf.linalg.matmul(obs_tp1['interest'], item_feats_tp1, transpose_b=True)
            score_next = v_all * q_tp1_online_all
            next_topk = tf.math.top_k(score_next, k=self._slate_size)
            next_topk_idx = next_topk.indices

            q_tp1_target_all, _ = self._target_q_network(obs_tp1, training=False)
            q_next_on_slate = tf.gather(q_tp1_target_all, next_topk_idx, axis=1, batch_dims=1)

            v_next_on_slate = tf.gather(v_all, next_topk_idx, axis=1, batch_dims=1)
            p_next = tf.nn.softmax(self._beta * v_next_on_slate, axis=-1)
            p_next = p_next * self._pos_w
            p_next = p_next / (tf.reduce_sum(p_next, axis=-1, keepdims=True) + 1e-8)

            expect_next = tf.reduce_sum(p_next * q_next_on_slate, axis=-1)

            r_scaled = reward_t / self._reward_scale
            y = tf.stop_gradient(r_scaled + self._gamma * discount_tp1 * expect_next)

            per_example = self._huber(y, q_clicked)
            per_example = per_example * tf.cast(clicked_mask, tf.float32)
            denom = tf.reduce_sum(tf.cast(clicked_mask, tf.float32)) + 1e-6
            loss = tf.reduce_sum(per_example) / denom

        grads = tape.gradient(loss, self._q_network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self._grad_clip_norm)
        self._optimizer.apply_gradients(zip(grads, self._q_network.trainable_variables))
        self._train_step_counter.assign_add(1)

        if self._tau > 0.0:
            for w_t, w in zip(self._target_q_network.weights, self._q_network.weights):
                w_t.assign(self._tau * w + (1.0 - self._tau) * w_t)
        else:
            step = int(self._train_step_counter.numpy())
            if step % self._target_update_period == 0:
                self._target_q_network.set_weights(self._q_network.get_weights())

        return tf_agent.LossInfo(loss=loss, extra={})