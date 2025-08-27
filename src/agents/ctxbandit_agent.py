import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib

from src.core.registry import register


class LinUCBScorer(tf.Module):
    """
    LinUCB scorer with ridge-regularized A = X^T X + λI and b = X^T r.
    Scores items with p + α * uncertainty, where p = x·θ and uncertainty = sqrt(x^T A^{-1} x).
    """
    def __init__(self, dim, alpha=1.0, ridge=1e-3, name="LinUCBScorer"):
        super().__init__(name=name)
        self._dim = int(dim)
        self.alpha = tf.Variable(float(alpha), trainable=False, dtype=tf.float32, name="alpha")
        self.ridge = float(ridge)
        self.A = tf.Variable(self.ridge * tf.eye(self._dim, dtype=tf.float32), trainable=False, name="A")
        self.b = tf.Variable(tf.zeros([self._dim], dtype=tf.float32), trainable=False, name="b")

    @tf.function
    def _theta(self):
        return tf.linalg.solve(self.A, self.b[:, None])[:, 0]

    @tf.function
    def scores(self, interest, item_feats_2d):
        theta = self._theta()
        X = interest * item_feats_2d
        p = tf.linalg.matvec(X, theta)
        invA = tf.linalg.inv(self.A)
        AX = tf.linalg.matvec(invA, X, transpose_a=False)          # [d] x [N,d]^T -> [N,d] via broadcasting
        u = tf.sqrt(tf.reduce_sum(X * AX, axis=1) + 1e-8)
        return p + self.alpha * u

    @tf.function
    def batch_update(self, X, r, mask):
        m = tf.cast(mask, tf.float32)[:, None]
        Xm = X * m
        rm = r * tf.squeeze(m, axis=1)
        A_delta = tf.einsum('bi,bj->ij', Xm, Xm)
        b_delta = tf.einsum('bi,b->i', Xm, rm)
        self.A.assign_add(A_delta)
        self.b.assign_add(b_delta)


class _GreedyTopKPolicy(tf_policy.TFPolicy):
    """
    Greedy top-K using LinUCB scores.
    """
    def __init__(self, time_step_spec, action_spec, scorer: LinUCBScorer, slate_size: int):
        super().__init__(time_step_spec, action_spec)
        self._scorer = scorer
        self._slate_size = int(slate_size)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation
        interest   = tf.convert_to_tensor(obs["interest"], tf.float32)
        item_feats = tf.convert_to_tensor(obs["item_features"], tf.float32)

        rank = tf.rank(item_feats)
        item_feats_2d = tf.case(
            [(tf.equal(rank, 3), lambda: item_feats[0])],
            default=lambda: item_feats
        )

        scores = self._scorer.scores(interest, item_feats_2d)
        top_k = tf.math.top_k(scores, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=top_k.indices)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class _EpsGreedyPolicy(tf_policy.TFPolicy):
    """
    Optional epsilon-greedy wrapper on top of LinUCB greedy policy.
    """
    def __init__(self, base_policy: _GreedyTopKPolicy, num_items: int, slate_size: int,
                 epsilon=0.0, steps_to_min=1, min_epsilon=0.0):
        super().__init__(base_policy.time_step_spec, base_policy.action_spec)
        self._base = base_policy
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self._epsilon = tf.Variable(float(epsilon), trainable=False, dtype=tf.float32)
        decay = (float(min_epsilon) / float(epsilon)) ** (1.0 / float(max(steps_to_min, 1))) if epsilon > 0 else 1.0
        self._epsilon_decay = tf.constant(decay, tf.float32)
        self._min_epsilon = tf.constant(float(min_epsilon), tf.float32)

    def _action(self, time_step, policy_state=(), seed=None):
        b = tf.shape(time_step.observation["interest"])[0]
        rand_scores = tf.random.uniform([b, self._num_items], dtype=tf.float32, seed=seed)
        random_slate = tf.math.top_k(rand_scores, k=self._slate_size).indices
        greedy_slate = self._base.action(time_step).action
        explore = tf.less(tf.random.uniform([b], dtype=tf.float32, seed=seed), self._epsilon)
        explore = tf.expand_dims(explore, 1)
        action = tf.where(explore, random_slate, greedy_slate)
        return policy_step.PolicyStep(action=action, state=policy_state)

    def decay_epsilon(self, steps=1):
        if steps <= 0:
            return
        new_eps = self._epsilon * tf.pow(self._epsilon_decay, steps)
        self._epsilon.assign(tf.maximum(self._min_epsilon, new_eps))

    @property
    def epsilon(self) -> float:
        return float(self._epsilon.numpy())


class CtxBanditAgent(tf_agent.TFAgent):
    """
    Contextual bandit with LinUCB updates; greedy top-K by UCB score; optional epsilon wrapper.
    """
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 alpha=1.0, ridge=1e-3,
                 epsilon=0.0, min_epsilon=0.0, epsilon_decay_steps=1):
        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self.is_learning = True

        dim = int(num_topics)
        self._scorer = LinUCBScorer(dim=dim, alpha=alpha, ridge=ridge)

        greedy = _GreedyTopKPolicy(time_step_spec, action_spec, self._scorer, slate_size)
        explore = _EpsGreedyPolicy(greedy, num_items, slate_size,
                                   epsilon=epsilon, steps_to_min=epsilon_decay_steps,
                                   min_epsilon=min_epsilon)

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=greedy,
            collect_policy=explore,
            train_sequence_length=2,
            train_step_counter=self._train_step_counter
        )

        self._huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)

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
        def slice_time(x, t_idx): return x[:, t_idx]

        def flatten_keep_tail(x, tail_ndims):
            x = tf.convert_to_tensor(x); shape = tf.shape(x); rank = tf.rank(x)
            def flatten_all(): return tf.reshape(x, [-1])
            def flatten_keep():
                if tail_ndims == 0: return tf.reshape(x, [-1])
                lead = tf.reduce_prod(shape[:-tail_ndims]); tail = shape[-tail_ndims:]
                return tf.reshape(x, tf.concat([[lead], tail], axis=0))
            return tf.cond(tf.less(rank, tail_ndims), flatten_all, flatten_keep)

        obs_t_interest   = slice_time(experience.observation['interest'], 0)
        click_pos_t      = slice_time(experience.observation['choice'],  1)
        reward_t         = slice_time(experience.reward,                  1)
        item_feats_t_any = slice_time(experience.observation['item_features'], 0)
        item_feats_rank  = tf.rank(item_feats_t_any)
        item_feats_t = tf.case(
            [(tf.equal(item_feats_rank, 4), lambda: item_feats_t_any[0, 0]),
             (tf.equal(item_feats_rank, 3), lambda: item_feats_t_any[0])],
            default=lambda: item_feats_t_any
        )

        act = experience.action
        act = act[:, 0] if tf.greater_equal(tf.rank(act), 2) else act

        interest_SB = flatten_keep_tail(obs_t_interest, tail_ndims=1)
        action_SB   = flatten_keep_tail(act, tail_ndims=1)
        click_SB    = tf.cast(flatten_keep_tail(click_pos_t, tail_ndims=0), tf.int32)
        reward_SB   = tf.cast(flatten_keep_tail(reward_t, tail_ndims=0), tf.float32)

        SB = tf.shape(action_SB)[0]
        slate_size = tf.shape(action_SB)[1]

        clicked_mask = tf.less(click_SB, slate_size)
        safe_click = tf.minimum(click_SB, slate_size - 1)
        idx = tf.stack([tf.range(SB), safe_click], axis=1)
        clicked_item_id = tf.gather_nd(action_SB, idx)

        clicked_feats_SB = tf.gather(item_feats_t, clicked_item_id)
        X = interest_SB * clicked_feats_SB

        self._scorer.batch_update(X, reward_SB, clicked_mask)

        theta = self._scorer._theta()
        pred = tf.linalg.matvec(X, theta)
        per_ex = self._huber(reward_SB, pred) * tf.cast(clicked_mask, tf.float32)
        denom = tf.reduce_sum(tf.cast(clicked_mask, tf.float32)) + 1e-6
        loss = tf.reduce_sum(per_ex) / denom

        self._train_step_counter.assign_add(1)
        return tf_agent.LossInfo(loss=loss, extra={})


@register("ctxbandit")
def _make_ctxbandit(time_step_spec, action_spec, **kw):
    return CtxBanditAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        num_users=kw.get("num_users", 10),
        num_topics=kw.get("num_topics", 10),
        slate_size=kw.get("slate_size", 5),
        num_items=kw.get("num_items", 100),
        alpha=kw.get("alpha", 1.0),
        ridge=kw.get("ridge", 1e-3),
        epsilon=kw.get("epsilon", 0.0),
        min_epsilon=kw.get("min_epsilon", 0.0),
        epsilon_decay_steps=kw.get("epsilon_decay_steps", 1),
    )
