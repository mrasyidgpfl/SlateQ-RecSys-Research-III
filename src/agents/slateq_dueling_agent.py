import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy


# ------------------------
# Utils
# ------------------------
def _safe(x):
    """Replace NaN/Inf with zeros to keep training stable."""
    x = tf.convert_to_tensor(x)
    return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

def _l2_norm(x, axis=-1, eps=1e-8):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return x / (tf.norm(x, ord=2, axis=axis, keepdims=True) + eps)


# ------------------------
# Networks / Policies
# ------------------------
class DuelingSlateQNetwork(network.Network):
    """Q(s,i) = V(s) + (A(s,i) - mean_i A(s,i))"""
    def __init__(self, input_tensor_spec, num_items: int, hidden=(256, 128, 64), l2=0.0,
                 name="DuelingSlateQNetwork"):
        super().__init__(input_tensor_spec={"interest": input_tensor_spec}, state_spec=(), name=name)
        self._num_items = int(num_items)
        reg = tf.keras.regularizers.l2(l2) if (l2 and l2 > 0) else None

        trunk = []
        for h in hidden:
            trunk.append(
                tf.keras.layers.Dense(
                    h, activation="relu",
                    kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
                    kernel_regularizer=reg,
                    dtype=tf.float32,
                )
            )
        self._trunk = tf.keras.Sequential(trunk, name=f"{name}_trunk")
        self._value = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=reg,
                                            name=f"{name}_value", dtype=tf.float32)
        self._advantage = tf.keras.layers.Dense(self._num_items, activation=None, kernel_regularizer=reg,
                                                name=f"{name}_advantage", dtype=tf.float32)

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = tf.convert_to_tensor(inputs["interest"], dtype=tf.float32)
        x = _safe(x)
        z = self._trunk(x, training=training)
        v = self._value(z, training=training)                   # [B, 1]
        a = self._advantage(z, training=training)               # [B, N]
        a_centered = a - tf.reduce_mean(a, axis=1, keepdims=True)
        q_values = v + a_centered                               # [B, N]
        return _safe(q_values), network_state


class SlateQPolicy(tf_policy.TFPolicy):
    """Greedy top-k slate by ranking v(s,i) * Q(s,i)."""
    def __init__(self, time_step_spec, action_spec, q_network, slate_size: int, beta: float = 2.0):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = int(slate_size)
        self._beta = float(beta)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation
        q_values, _ = self._q_network(obs, time_step.step_type)   # [B, N]

        interest = tf.convert_to_tensor(obs["interest"], dtype=tf.float32)        # [B, T]
        item_feats = tf.convert_to_tensor(obs["item_features"], dtype=tf.float32) # [B?, N, T] or [N, T]
        # Ensure [B, N, T]
        item_feats = tf.cond(
            tf.equal(tf.rank(item_feats), 2),
            lambda: tf.tile(item_feats[None, ...], [tf.shape(interest)[0], 1, 1]),
            lambda: item_feats
        )

        # Light normalisation + stop_gradient (they are mixture weights only)
        i_n = tf.stop_gradient(_l2_norm(interest, axis=-1))
        f_n = tf.stop_gradient(_l2_norm(item_feats, axis=-1))
        v_all = _safe(tf.einsum("bt,bnt->bn", i_n, f_n))          # [B, N]

        score = _safe(v_all * q_values)                           # [B, N]
        top_k = tf.math.top_k(score, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=tf.cast(top_k.indices, tf.int32))

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class SlateQExplorationPolicy(tf_policy.TFPolicy):
    """Epsilon-greedy wrapper with exponential decay."""
    def __init__(self, base_policy: SlateQPolicy, num_items: int, slate_size: int,
                 epsilon: float = 0.2, steps_to_min: int = 20_000, min_epsilon: float = 0.05):
        super().__init__(base_policy.time_step_spec, base_policy.action_spec)
        self._base_policy = base_policy
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self._epsilon = tf.Variable(float(epsilon), trainable=False, dtype=tf.float32)
        decay = (float(min_epsilon) / float(epsilon)) ** (1.0 / float(steps_to_min))
        self._epsilon_decay = tf.constant(decay, dtype=tf.float32)
        self._min_epsilon = tf.constant(float(min_epsilon), dtype=tf.float32)

    def _action(self, time_step, policy_state=(), seed=None):
        bsz = tf.shape(time_step.observation["interest"])[0]
        rand_scores = tf.random.uniform([bsz, self._num_items], dtype=tf.float32, seed=seed)
        random_slate = tf.math.top_k(rand_scores, k=self._slate_size).indices
        greedy_slate = self._base_policy.action(time_step).action
        explore = tf.less(tf.random.uniform([bsz], dtype=tf.float32, seed=seed), self._epsilon)
        explore = tf.expand_dims(explore, 1)
        action = tf.where(explore, random_slate, greedy_slate)
        return policy_step.PolicyStep(action=tf.cast(action, tf.int32), state=policy_state)

    def decay_epsilon(self, steps=1):
        if steps <= 0:
            return
        new_eps = self._epsilon * tf.pow(self._epsilon_decay, steps)
        self._epsilon.assign(tf.maximum(self._min_epsilon, new_eps))

    @property
    def epsilon(self) -> float:
        return float(self._epsilon.numpy())


# ------------------------
# Agent
# ------------------------
class SlateQDuelingAgent(tf_agent.TFAgent):
    """
    Dueling SlateQ with:
      • expected next-slate value, Double-Q targets
      • reward squashing → (tanh(r)+1)/2 in [0,1]
      • non-click backup: E_t[Q(s,a)|slate]
      • grad clipping, Huber loss, Polyak target updates
      • lean replay support (optional)
    """
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=1e-3,
                 epsilon=0.2, min_epsilon=0.05, epsilon_decay_steps=20_000,
                 target_update_period=1000, tau=0.003, gamma=0.95, beta=2.0,
                 huber_delta=1.0, grad_clip_norm=10.0, reward_scale=10.0,
                 l2=0.0, pos_weights=None):
        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self._target_update_period = int(target_update_period)
        self._tau = float(tau)
        self._gamma = tf.constant(gamma, dtype=tf.float32)
        self._beta = tf.constant(beta, dtype=tf.float32)
        self._grad_clip_norm = float(grad_clip_norm)
        self._reward_scale = float(reward_scale)  # kept for API compat
        self.is_learning = True

        # Optional lean replay: cache static [N, T] features
        self._static_item_features = None

        # Position weights
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

        # Networks
        input_tensor_spec = time_step_spec.observation["interest"]
        self._q_network = DuelingSlateQNetwork(input_tensor_spec, num_items, l2=l2)
        self._target_q_network = DuelingSlateQNetwork(input_tensor_spec, num_items, l2=l2)

        # Build once and copy weights
        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network.set_weights(self._q_network.get_weights())

        # Optimiser / loss
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._huber = tf.keras.losses.Huber(delta=huber_delta,
                                            reduction=tf.keras.losses.Reduction.NONE)

        # Policies
        base_policy = SlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size,
            beta=float(self._beta.numpy()) if hasattr(self._beta, "numpy") else self._beta,
        )
        self._policy = base_policy
        self._collect_policy = SlateQExplorationPolicy(
            base_policy=base_policy,
            num_items=num_items,
            slate_size=slate_size,
            epsilon=epsilon,
            steps_to_min=epsilon_decay_steps,
            min_epsilon=min_epsilon
        )

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=self._policy,
            collect_policy=self._collect_policy,
            train_sequence_length=2,
            train_step_counter=self._train_step_counter
        )

    # ---- lean replay hook (optional) ----
    def set_static_item_features(self, item_features_nt: tf.Tensor):
        x = tf.convert_to_tensor(item_features_nt, dtype=tf.float32)
        if x.shape.rank == 3:
            x = x[0]
        self._static_item_features = _safe(x)

    @property
    def collect_data_spec(self):
        if self._static_item_features is not None:
            # Lean spec: no item_features in replay
            return trajectory_lib.Trajectory(
                step_type=self._time_step_spec.step_type,
                observation={
                    "interest": self._time_step_spec.observation["interest"],
                    "choice": self._time_step_spec.observation["choice"],
                },
                action=self._action_spec,
                policy_info=(),
                next_step_type=self._time_step_spec.step_type,
                reward=self._time_step_spec.reward,
                discount=self._time_step_spec.discount,
            )
        # Full spec
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

    # ------------------------
    # Training
    # ------------------------
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

        def align_leading(x, B):
            x = tf.convert_to_tensor(x)
            xB = tf.shape(x)[0]
            def same():
                return x
            def repeat_or_trunc():
                idx = tf.math.mod(tf.range(B), xB)
                return tf.gather(x, idx, axis=0)
            return tf.cond(tf.equal(xB, B), same, repeat_or_trunc)

        def to_BNT(feats_any, B, N, T):
            rnk = tf.rank(feats_any)
            def from4():
                x = flatten_keep_tail(feats_any, tail_ndims=2)
                return align_leading(x, B)
            def from3():
                x = flatten_keep_tail(feats_any, tail_ndims=2)
                return align_leading(x, B)
            def from2():
                x = tf.expand_dims(feats_any, 0)
                return tf.tile(x, [B, 1, 1])
            x = tf.case([(tf.equal(rnk, 4), from4), (tf.equal(rnk, 3), from3)], default=from2)
            return tf.reshape(x, [B, N, T])

        # ---- unpack t and t+1 ----
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

        click_pos_any  = slice_time(experience.observation["choice"], 1)
        click_pos_flat = flatten_keep_tail(click_pos_any, tail_ndims=0)
        click_pos_t    = tf.cast(align_leading(click_pos_flat, B), tf.int32)

        # item features t / t+1 → [B, N, T]
        if self._static_item_features is not None:
            item_feats_t   = tf.tile(self._static_item_features[None, ...], [B, 1, 1])
            item_feats_tp1 = item_feats_t
        else:
            item_feats_t_any   = slice_time(experience.observation["item_features"], 0)
            item_feats_tp1_any = slice_time(experience.observation["item_features"], 1)
            item_feats_t   = to_BNT(item_feats_t_any,   B, N, T)
            item_feats_tp1 = to_BNT(item_feats_tp1_any, B, N, T)

        # ---- click handling ----
        batch_size = tf.shape(action_t)[0]
        slate_size = tf.shape(action_t)[1]
        clicked_mask = tf.less(click_pos_t, slate_size)
        safe_click_pos = tf.minimum(click_pos_t, slate_size - 1)
        row_idx = tf.range(batch_size, dtype=tf.int32)
        idx = tf.stack([row_idx, safe_click_pos], axis=1)
        clicked_item_id = tf.gather_nd(action_t, idx)  # [B]

        with tf.GradientTape() as tape:
            # Q(s_t, ·)
            q_tm1_all, _ = self._q_network(obs_t, training=True)                 # [B, N]
            q_tm1_all = _safe(q_tm1_all)
            q_clicked = tf.gather(q_tm1_all, clicked_item_id, axis=1, batch_dims=1)  # [B]

            # Expected Q over shown slate at time t (for non-clicks)
            q_on_slate_t = tf.gather(q_tm1_all, action_t, axis=1, batch_dims=1)      # [B, K]

            # Affinity-based mixture weights (stop grad; clip logits)
            i_t_n  = tf.stop_gradient(_l2_norm(obs_t["interest"], axis=-1))
            ft_n   = tf.stop_gradient(_l2_norm(item_feats_t, axis=-1))
            v_all_t = _safe(tf.einsum("bt,bnt->bn", i_t_n, ft_n))                    # [B, N]
            v_on_slate_t = tf.gather(v_all_t, action_t, axis=1, batch_dims=1)        # [B, K]
            logits_t = tf.clip_by_value(self._beta * v_on_slate_t, -15.0, 15.0)
            p_t = tf.nn.softmax(logits_t, axis=-1)                                   # [B, K]
            p_t = p_t * self._pos_w
            p_t = p_t / (tf.reduce_sum(p_t, axis=-1, keepdims=True) + 1e-8)
            q_expected_t = tf.reduce_sum(p_t * q_on_slate_t, axis=-1)                # [B]

            # ---- Double-Q target on next state ----
            q_tp1_online_all, _ = self._q_network(obs_tp1, training=False)           # [B, N]
            q_tp1_online_all = _safe(q_tp1_online_all)

            i_tp1_n  = tf.stop_gradient(_l2_norm(obs_tp1["interest"], axis=-1))
            f_tp1_n  = tf.stop_gradient(_l2_norm(item_feats_tp1, axis=-1))
            v_all_tp1 = _safe(tf.einsum("bt,bnt->bn", i_tp1_n, f_tp1_n))             # [B, N]

            score_next = _safe(v_all_tp1 * q_tp1_online_all)                         # [B, N]
            next_topk_idx = tf.math.top_k(score_next, k=self._slate_size).indices    # [B, K]

            q_tp1_target_all, _ = self._target_q_network(obs_tp1, training=False)    # [B, N]
            q_tp1_target_all = _safe(q_tp1_target_all)

            q_next_on_slate   = tf.gather(q_tp1_target_all, next_topk_idx, axis=1, batch_dims=1)  # [B, K]
            v_next_on_slate   = tf.gather(v_all_tp1,        next_topk_idx, axis=1, batch_dims=1)  # [B, K]
            logits_next = tf.clip_by_value(self._beta * v_next_on_slate, -15.0, 15.0)
            p_next = tf.nn.softmax(logits_next, axis=-1)
            p_next = p_next * self._pos_w
            p_next = p_next / (tf.reduce_sum(p_next, axis=-1, keepdims=True) + 1e-8)

            expect_next = tf.reduce_sum(p_next * q_next_on_slate, axis=-1)           # [B]

            # ---- rewards & discount (targets in [0,1]) ----
            reward_t_any      = slice_time(experience.reward, 1)
            discount_tp1_any  = tf.cast(slice_time(experience.discount, 1), tf.float32)
            reward_flat       = flatten_keep_tail(reward_t_any,     tail_ndims=0)
            discount_flat     = flatten_keep_tail(discount_tp1_any, tail_ndims=0)
            r_raw             = align_leading(reward_flat, B)
            r_scaled          = _safe((tf.tanh(r_raw) + 1.0) * 0.5)
            discount_tp1      = tf.clip_by_value(align_leading(discount_flat, B), 0.0, 1.0)

            y = tf.stop_gradient(r_scaled + self._gamma * discount_tp1 * expect_next)  # [B]

            # Predictive value at time t
            q_pred_t = _safe(tf.where(clicked_mask, q_clicked, q_expected_t))          # [B]

            per_example = self._huber(y, q_pred_t)
            per_example = _safe(per_example)
            loss = tf.reduce_mean(per_example)
            loss = _safe(loss)

        grads = tape.gradient(loss, self._q_network.trainable_variables)
        # Guard against NaN grads
        grads = [
            (tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None)
            for g in grads
        ]
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
