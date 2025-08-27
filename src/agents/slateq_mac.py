import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import mixed_precision

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy


class VanillaSlateQNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 num_items: int,
                 hidden=(512, 256, 128),
                 l2=1e-5,
                 name='VanillaSlateQNetwork'):
        super().__init__(input_tensor_spec={'interest': input_tensor_spec},
                         state_spec=(),
                         name=name)
        self._num_items = int(num_items)
        reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

        layers = []
        for h in hidden:
            layers.append(
                tf.keras.layers.Dense(
                    h,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
                    kernel_regularizer=reg
                )
            )
            layers.append(tf.keras.layers.LayerNormalization())
            layers.append(tf.keras.layers.Activation('relu'))
        self._mlp = tf.keras.Sequential(layers, name=f"{name}_mlp")
        self._head = tf.keras.layers.Dense(
            self._num_items,
            activation=None,
            kernel_regularizer=reg,
            dtype="float32",
            name=f"{name}_head"
        )

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = tf.convert_to_tensor(inputs['interest'])
        x = self._mlp(x, training=training)
        q_values = self._head(x, training=training)
        q_values = tf.cast(q_values, tf.float32)
        return q_values, network_state


class SlateQPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, q_network, slate_size: int, beta: float = 7.5):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = int(slate_size)
        self._beta = float(beta)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation

        q_values, _ = self._q_network(obs, time_step.step_type)
        q_values = tf.cast(q_values, tf.float32)

        interest = tf.cast(tf.convert_to_tensor(obs["interest"]), tf.float32)
        item_feats = tf.cast(tf.convert_to_tensor(obs["item_features"]), tf.float32)

        item_feats = tf.cond(
            tf.equal(tf.rank(item_feats), 2),
            lambda: tf.tile(tf.expand_dims(item_feats, 0), [tf.shape(interest)[0], 1, 1]),
            lambda: item_feats
        )

        v_all = tf.einsum("bt,bnt->bn", interest, item_feats)
        v_all = tf.cast(v_all, tf.float32)

        score = v_all * q_values
        top_k = tf.math.top_k(score, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=top_k.indices)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class SlateQExplorationPolicy(tf_policy.TFPolicy):
    def __init__(self,
                 base_policy: SlateQPolicy,
                 num_items: int,
                 slate_size: int,
                 epsilon: float = 0.15,
                 steps_to_min: int = 20_000,
                 min_epsilon: float = 0.03):
        super().__init__(base_policy.time_step_spec, base_policy.action_spec)
        self._base_policy = base_policy
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)

        self._epsilon = tf.Variable(float(epsilon), trainable=False, dtype=tf.float32)
        decay = (float(min_epsilon) / float(epsilon)) ** (1.0 / float(steps_to_min))
        self._epsilon_decay = tf.constant(decay, dtype=tf.float32)
        self._min_epsilon = tf.constant(float(min_epsilon), dtype=tf.float32)

    def _action(self, time_step, policy_state=(), seed=None):
        batch_size = tf.shape(time_step.observation['interest'])[0]

        rand_scores = tf.random.uniform([batch_size, self._num_items], dtype=tf.float32, seed=seed)
        random_slate = tf.math.top_k(rand_scores, k=self._slate_size).indices

        greedy_slate = self._base_policy.action(time_step).action

        explore_mask = tf.less(
            tf.random.uniform([batch_size], dtype=tf.float32, seed=seed),
            self._epsilon
        )
        explore_mask = tf.expand_dims(explore_mask, 1)

        action = tf.where(explore_mask, random_slate, greedy_slate)
        return policy_step.PolicyStep(action=action, state=policy_state)

    def decay_epsilon(self, steps=1):
        if steps <= 0:
            return
        new_eps = self._epsilon * tf.pow(self._epsilon_decay, steps)
        self._epsilon.assign(tf.maximum(self._min_epsilon, new_eps))

    @property
    def epsilon(self) -> float:
        return float(self._epsilon.numpy())


class SlateQAgent(tf_agent.TFAgent):
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=2e-4,
                 epsilon=0.15, min_epsilon=0.03, epsilon_decay_steps=20_000,
                 target_update_period=1000,
                 tau=0.01,
                 gamma=0.98,
                 beta=7.5,
                 huber_delta=2.0,
                 grad_clip_norm=5.0,
                 reward_scale=8.0,
                 l2=1e-5,
                 pos_weights=(1.0, 0.75, 0.55, 0.40, 0.30)):
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

        self._q_network = VanillaSlateQNetwork(input_tensor_spec, num_items, l2=l2)
        self._target_q_network = VanillaSlateQNetwork(input_tensor_spec, num_items, l2=l2)

        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network.set_weights(self._q_network.get_weights())

        base_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._optimizer = mixed_precision.LossScaleOptimizer(base_opt)
        self._huber = tf.keras.losses.Huber(delta=huber_delta,
                                            reduction=tf.keras.losses.Reduction.NONE)

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
        item_feats_tp1 = tf.cast(item_feats_tp1, tf.float32)

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
        obs_t['interest'] = tf.cast(obs_t['interest'], tf.float32)
        obs_tp1['interest'] = tf.cast(obs_tp1['interest'], tf.float32)
        action_t     = flatten_keep_tail(action_t,     tail_ndims=1)
        click_pos_t  = tf.cast(flatten_keep_tail(click_pos_t,  tail_ndims=0), tf.int32)
        reward_t     = tf.cast(flatten_keep_tail(reward_t,     tail_ndims=0), tf.float32)
        discount_tp1 = flatten_keep_tail(discount_tp1, tail_ndims=0)

        batch_size = tf.shape(action_t)[0]
        slate_size = tf.shape(action_t)[1]

        clicked_mask = tf.less(click_pos_t, slate_size)
        safe_click_pos = tf.minimum(click_pos_t, slate_size - 1)
        idx = tf.stack([tf.range(batch_size), safe_click_pos], axis=1)
        clicked_item_id = tf.gather_nd(action_t, idx)

        with tf.GradientTape() as tape:
            q_tm1_all, _ = self._q_network(obs_t, training=True)
            q_tm1_all = tf.cast(q_tm1_all, tf.float32)
            q_clicked = tf.gather(q_tm1_all, clicked_item_id, axis=1, batch_dims=1)

            q_tp1_online_all, _ = self._q_network(obs_tp1, training=False)
            q_tp1_online_all = tf.cast(q_tp1_online_all, tf.float32)

            v_all = tf.linalg.matmul(obs_tp1['interest'], item_feats_tp1, transpose_b=True)
            v_all = tf.cast(v_all, tf.float32)

            score_next = v_all * q_tp1_online_all
            next_topk = tf.math.top_k(score_next, k=self._slate_size)
            next_topk_idx = next_topk.indices

            q_tp1_target_all, _ = self._target_q_network(obs_tp1, training=False)
            q_tp1_target_all = tf.cast(q_tp1_target_all, tf.float32)
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

            scaled_loss = self._optimizer.get_scaled_loss(loss)

        grads = tape.gradient(scaled_loss, self._q_network.trainable_variables)
        grads = self._optimizer.get_unscaled_gradients(grads)
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
