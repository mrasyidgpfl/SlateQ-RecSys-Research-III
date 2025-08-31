import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy


def _safe(x, name=None):
    """Replace NaN/Inf with zeros to keep training stable."""
    x = tf.convert_to_tensor(x)
    return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x), name=name)

def _l2_norm(x, axis=-1, eps=1e-8):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return x / (tf.norm(x, ord=2, axis=axis, keepdims=True) + eps)


class NoisyDense(tf.keras.layers.Layer):
    """Factorised Gaussian NoisyNet layer (Fortunato et al., 2017)."""
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
        return tf.sign(eps) * tf.sqrt(tf.abs(eps) + 1e-8)

    def _sample_noise(self):
        eps_in = tf.random.normal([self._in_features])
        eps_out = tf.random.normal([self.units])
        f_in = NoisyDense._f(eps_in)
        f_out = NoisyDense._f(eps_out)
        w_noise = tf.einsum("i,j->ij", f_in, f_out)
        b_noise = f_out if self.use_bias else None
        return w_noise, b_noise

    def call(self, x, training=False):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
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


class DuelingNoisySlateQNetwork(network.Network):
    def __init__(self, input_tensor_spec, num_items: int,
                 hidden=(256, 128, 64), sigma0=0.5, l2=0.0, name="DuelingNoisySlateQNetwork"):
        super().__init__(input_tensor_spec={"interest": input_tensor_spec}, state_spec=(), name=name)
        self._num_items = int(num_items)
        reg = tf.keras.regularizers.l2(l2) if (l2 and l2 > 0) else None

        layers = [NoisyDense(h, sigma0=sigma0, activation="relu", kernel_regularizer=reg) for h in hidden]
        self._trunk = tf.keras.Sequential(layers, name=f"{name}_trunk")

        self._value = NoisyDense(1, sigma0=sigma0, activation=None, kernel_regularizer=reg, name=f"{name}_value")
        self._advantage = NoisyDense(self._num_items, sigma0=sigma0, activation=None, kernel_regularizer=reg,
                                     name=f"{name}_advantage")

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = _safe(tf.convert_to_tensor(inputs["interest"], dtype=tf.float32))
        z = self._trunk(x, training=training)
        v = self._value(z, training=training)
        a = self._advantage(z, training=training)
        a_centered = a - tf.reduce_mean(a, axis=1, keepdims=True)
        q_values = _safe(v + a_centered)
        return q_values, network_state


class NoisyAwareSlateQPolicy(tf_policy.TFPolicy):
    """Greedy top-k slate by ranking (cosine-style affinity) * Q(s,i)."""
    def __init__(self, time_step_spec, action_spec, q_network, slate_size: int, beta: float = 2.0, use_noisy: bool = True):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = int(slate_size)
        self._beta = float(beta)
        self._use_noisy = bool(use_noisy)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation
        q_values, _ = self._q_network(obs, time_step.step_type, training=self._use_noisy)

        interest = tf.convert_to_tensor(obs["interest"], dtype=tf.float32)
        item_feats = tf.convert_to_tensor(obs["item_features"], dtype=tf.float32)
        item_feats = tf.cond(
            tf.equal(tf.rank(item_feats), 2),
            lambda: tf.tile(item_feats[None, ...], [tf.shape(interest)[0], 1, 1]),
            lambda: item_feats
        )

        i_n = _l2_norm(interest, axis=-1)
        f_n = _l2_norm(item_feats, axis=-1)
        v_all = _safe(tf.einsum("bt,bnt->bn", i_n, f_n))

        score = _safe(v_all * q_values)
        top_k = tf.math.top_k(score, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=tf.cast(top_k.indices, tf.int32))

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class SlateQDuelingNoisyNetAgent(tf_agent.TFAgent):
    """
    SlateQ + Dueling + NoisyNet with stability fixes:
    - Expected next-slate value (position-weighted) + Double-Q targets
    - Reward squashing to [0,1]: (tanh(r) + 1) / 2
    - Back-up on non-clicks with expected Q over shown slate
    - Huber loss, NaN-grad guard, global-norm clip, Polyak target updates
    - LEAN replay: cache static features via set_static_item_features()
    """
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=1e-3,
                 target_update_period=1000, tau=0.003, gamma=0.95, beta=2.0,
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
        self._reward_scale = float(reward_scale)
        self.is_learning = True

        self._static_item_features = None

        if pos_weights is None:
            pw = tf.linspace(1.0, 0.3, self._slate_size)
        else:
            pw = tf.convert_to_tensor(pos_weights, dtype=tf.float32)
            cur = tf.shape(pw)[0]
            def pad():
                pad_len = self._slate_size - cur
                return tf.concat([pw, tf.fill([pad_len], pw[-1])], axis=0)
            def trunc():
                return pw[:self._slate_size]
            pw = tf.cond(cur < self._slate_size, pad, trunc)
        self._pos_w = tf.reshape(pw, [1, self._slate_size])

        input_tensor_spec = time_step_spec.observation["interest"]
        self._q_network = DuelingNoisySlateQNetwork(input_tensor_spec, num_items, sigma0=noisy_sigma0, l2=l2)
        self._target_q_network = DuelingNoisySlateQNetwork(input_tensor_spec, num_items, sigma0=noisy_sigma0, l2=l2)

        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network.set_weights(self._q_network.get_weights())

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)

        self._policy = NoisyAwareSlateQPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            q_network=self._q_network, slate_size=slate_size,
            beta=float(self._beta.numpy()) if hasattr(self._beta, "numpy") else self._beta,
            use_noisy=bool(noisy_eval_eval),
        )
        self._collect_policy = NoisyAwareSlateQPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            q_network=self._q_network, slate_size=slate_size,
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

    def set_static_item_features(self, item_features_nt: tf.Tensor):
        """Cache a static [N,T] matrix to enable lean replay (no item_features in each transition)."""
        x = tf.convert_to_tensor(item_features_nt, dtype=tf.float32)
        if x.shape.rank == 3:
            x = x[0]
        self._static_item_features = _safe(x)

    @property
    def collect_data_spec(self):
        if self._static_item_features is not None:
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
