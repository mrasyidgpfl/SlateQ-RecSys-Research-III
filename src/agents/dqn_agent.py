import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy


# -----------------------------
# Per-item Q network
# -----------------------------
class DQNNetwork(network.Network):
    """
    Feed-forward Q-network mapping interest features to per-item Q-values.
    """
    def __init__(self,
                 input_tensor_spec,
                 num_items: int,
                 hidden=(256, 128, 64),
                 l2=0.0,
                 name='DQNNetwork'):
        super().__init__(input_tensor_spec={'interest': input_tensor_spec},
                         state_spec=(),
                         name=name)
        self._num_items = int(num_items)
        reg = tf.keras.regularizers.l2(l2) if (l2 and l2 > 0) else None

        layers = []
        for h in hidden:
            layers.append(
                tf.keras.layers.Dense(
                    h,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
                    kernel_regularizer=reg
                )
            )
        self._mlp = tf.keras.Sequential(layers, name=f"{name}_mlp")
        self._head = tf.keras.layers.Dense(
            self._num_items,
            activation=None,
            kernel_regularizer=reg,
            name=f"{name}_head"
        )

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = tf.convert_to_tensor(inputs['interest'])
        x = self._mlp(x, training=training)
        q_values = self._head(x, training=training)
        return q_values, network_state


# -----------------------------
# Greedy policy
# -----------------------------
class DQNGreedyPolicy(tf_policy.TFPolicy):
    """
    Greedy policy over per-item Q-values.

    If action_spec is a slate (shape [K]), rank items by v(s,i) * Q(s,i) and take top-K.
    If action_spec is scalar, pick argmax(Q).
    """
    def __init__(self, time_step_spec, action_spec, q_network):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._is_slate = (len(action_spec.shape) == 1)
        self._slate_size = int(action_spec.shape[0]) if self._is_slate else 1

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation
        q_values, _ = self._q_network(obs, time_step.step_type)  # [B, N]

        if self._is_slate:
            # Compute user-item affinity v(s,i) = <interest, item_features_i>
            interest   = tf.convert_to_tensor(obs["interest"])        # [B, T]
            item_feats = tf.convert_to_tensor(obs["item_features"])   # [B, N, T] or [N, T]

            # Ensure [B, N, T]
            item_feats = tf.cond(
                tf.equal(tf.rank(item_feats), 2),
                lambda: tf.tile(item_feats[None, ...], [tf.shape(interest)[0], 1, 1]),
                lambda: item_feats
            )

            v_all = tf.einsum("bt,bnt->bn", interest, item_feats)  # [B, N]
            score = v_all * q_values                                # [B, N]
            loc = tf.math.top_k(score, k=self._slate_size).indices  # [B, K]
        else:
            loc = tf.argmax(q_values, axis=-1, output_type=tf.int32)  # [B]

        return tfp.distributions.Deterministic(loc=loc)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


# -----------------------------
# Epsilon-greedy wrapper
# -----------------------------
class DQNExplorationPolicy(tf_policy.TFPolicy):
    """
    Epsilon-greedy wrapper around DQNGreedyPolicy, with exponential decay.
    """
    def __init__(self,
                 base_policy: DQNGreedyPolicy,
                 num_items: int,
                 epsilon: float = 0.2,
                 steps_to_min: int = 20_000,
                 min_epsilon: float = 0.05):
        super().__init__(base_policy.time_step_spec, base_policy.action_spec)
        self._base = base_policy
        self._num_items = int(num_items)
        self._is_slate = (len(self._action_spec.shape) == 1)
        self._slate_size = int(self._action_spec.shape[0]) if self._is_slate else 1

        self._epsilon = tf.Variable(float(epsilon), trainable=False, dtype=tf.float32)
        decay = (float(min_epsilon) / float(epsilon)) ** (1.0 / float(steps_to_min))
        self._epsilon_decay = tf.constant(decay, dtype=tf.float32)
        self._min_epsilon = tf.constant(float(min_epsilon), dtype=tf.float32)

    def _action(self, time_step, policy_state=(), seed=None):
        batch = tf.shape(time_step.observation['interest'])[0]
        if self._is_slate:
            rand_scores = tf.random.uniform([batch, self._num_items], dtype=tf.float32, seed=seed)
            rand_act = tf.math.top_k(rand_scores, k=self._slate_size).indices  # [B, K]
        else:
            rand_act = tf.random.uniform([batch], minval=0, maxval=self._num_items,
                                         dtype=tf.int32, seed=seed)            # [B]

        greedy_act = self._base.action(time_step).action
        explore = tf.less(tf.random.uniform([batch], dtype=tf.float32, seed=seed), self._epsilon)
        explore = tf.expand_dims(explore, 1) if self._is_slate else explore
        act = tf.where(explore, rand_act, greedy_act)
        return policy_step.PolicyStep(action=act, state=policy_state)

    def decay_epsilon(self, steps=1):
        if steps <= 0:
            return
        new_eps = self._epsilon * tf.pow(self._epsilon_decay, steps)
        self._epsilon.assign(tf.maximum(self._min_epsilon, new_eps))

    @property
    def epsilon(self) -> float:
        return float(self._epsilon.numpy())


# -----------------------------
# DQN Agent (with Double DQN target)
# -----------------------------
class DQNAgent(tf_agent.TFAgent):
    """
    DQN agent over items. Works with scalar or slate actions.
    Uses:
      - v(s,i)*Q(s,i) ranking for slate selection
      - Double DQN targets
      - Click masking, reward scaling, gradient clipping
    """
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, num_items=100,
                 learning_rate=1e-3,
                 epsilon=0.2, min_epsilon=0.05, epsilon_decay_steps=20_000,
                 target_update_period=1000,
                 tau=0.005,
                 gamma=0.95,
                 huber_delta=1.0,
                 grad_clip_norm=10.0,
                 reward_scale=10.0,
                 l2=0.0):
        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._target_update_period = int(target_update_period)
        self._tau = float(tau)
        self._gamma = tf.constant(gamma, dtype=tf.float32)
        self._grad_clip_norm = float(grad_clip_norm)
        self._reward_scale = float(reward_scale)
        self.is_learning = True

        input_tensor_spec = time_step_spec.observation['interest']
        self._q_net = DQNNetwork(input_tensor_spec, num_items, l2=l2)
        self._tgt_net = DQNNetwork(input_tensor_spec, num_items, l2=l2)

        # Build nets
        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_net(dummy_obs, dummy_step, training=False)
        self._tgt_net(dummy_obs, dummy_step, training=False)
        self._tgt_net.set_weights(self._q_net.get_weights())

        self._opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._huber = tf.keras.losses.Huber(delta=huber_delta,
                                            reduction=tf.keras.losses.Reduction.NONE)

        base = DQNGreedyPolicy(time_step_spec, action_spec, self._q_net)
        self._policy = base
        self._collect_policy = DQNExplorationPolicy(
            base_policy=base,
            num_items=num_items,
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

        # ----- slice and normalize shapes -----
        obs_t_interest   = slice_time(experience.observation['interest'], 0)   # [*, T]
        obs_tp1_interest = slice_time(experience.observation['interest'], 1)   # [*, T]
        click_pos_t      = slice_time(experience.observation['choice'], 1)     # [*]

        act_t = slice_time(experience.action, 0)  # [*, K] or [*]
        act_t_rank = tf.rank(act_t)
        action_t = tf.cond(
            tf.equal(act_t_rank, 1),
            lambda: tf.expand_dims(act_t, -1),  # [*, 1]
            lambda: act_t                        # [*, K]
        )

        reward_t      = slice_time(experience.reward, 1)     # [*]
        discount_tp1  = tf.cast(slice_time(experience.discount, 1), tf.float32)

        obs_t   = {'interest': flatten_keep_tail(obs_t_interest,   tail_ndims=1)}  # [B, T]
        obs_tp1 = {'interest': flatten_keep_tail(obs_tp1_interest, tail_ndims=1)}  # [B, T]
        action_t     = flatten_keep_tail(action_t,     tail_ndims=1)               # [B, K]
        click_pos_t  = tf.cast(flatten_keep_tail(click_pos_t,  tail_ndims=0), tf.int32)  # [B]
        reward_t     = flatten_keep_tail(reward_t,     tail_ndims=0)               # [B]
        discount_tp1 = flatten_keep_tail(discount_tp1, tail_ndims=0)               # [B]

        batch_size = tf.shape(action_t)[0]
        slate_size = tf.shape(action_t)[1]

        clicked_mask = tf.less(click_pos_t, slate_size)           # [B]
        safe_click_pos = tf.minimum(click_pos_t, slate_size - 1)  # [B]

        row_idx = tf.range(batch_size, dtype=tf.int32)            # [B]
        clicked_item_id = tf.gather_nd(action_t, tf.stack([row_idx, safe_click_pos], axis=1))  # [B]

        # ----- loss -----
        with tf.GradientTape() as tape:
            # Q(s,a_clicked)
            q_tm1_all, _ = self._q_net(obs_t, training=True)  # [B, N]
            q_clicked = tf.gather_nd(q_tm1_all, tf.stack([row_idx, clicked_item_id], axis=1))  # [B]

            # Double DQN target: argmax from online, value from target
            q_tp1_online, _ = self._q_net(obs_tp1, training=False)        # [B, N]
            a_next = tf.argmax(q_tp1_online, axis=-1, output_type=tf.int32)  # [B]
            q_tp1_tgt, _    = self._tgt_net(obs_tp1, training=False)      # [B, N]
            q_next_max = tf.gather_nd(q_tp1_tgt, tf.stack([row_idx, a_next], axis=1))  # [B]

            r_scaled = reward_t / self._reward_scale
            y = tf.stop_gradient(r_scaled + self._gamma * discount_tp1 * q_next_max)  # [B]

            per_ex = self._huber(y, q_clicked)                                     # [B]
            per_ex = per_ex * tf.cast(clicked_mask, tf.float32)
            denom = tf.reduce_sum(tf.cast(clicked_mask, tf.float32)) + 1e-6
            loss = tf.reduce_sum(per_ex) / denom

        grads = tape.gradient(loss, self._q_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self._grad_clip_norm)
        self._opt.apply_gradients(zip(grads, self._q_net.trainable_variables))
        self._train_step_counter.assign_add(1)

        # Target update: soft if tau>0 else periodic hard
        if self._tau > 0.0:
            for w_t, w in zip(self._tgt_net.weights, self._q_net.weights):
                w_t.assign(self._tau * w + (1.0 - self._tau) * w_t)
        else:
            step = int(self._train_step_counter.numpy())
            if step % self._target_update_period == 0:
                self._tgt_net.set_weights(self._q_net.get_weights())

        return tf_agent.LossInfo(loss=loss, extra={})
