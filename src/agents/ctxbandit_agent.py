import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib

from src.core.registry import register


class _LinearScorer(tf.Module):
    def __init__(self, dim, l2=0.0, name="LinearScorer"):
        super().__init__(name=name)
        self.w = tf.Variable(tf.zeros([dim], tf.float32), name="w")
        self.b = tf.Variable(0.0, tf.float32, name="b")
        self._l2 = float(l2)

    def scores(self, interest, item_feats_2d):
        iw = interest * self.w                    
        return tf.linalg.matmul(iw, item_feats_2d, transpose_b=True) + self.b

    def clicked_pred(self, interest_b, item_feat_clicked_b):
        z = interest_b * item_feat_clicked_b
        return tf.reduce_sum(z * self.w, axis=-1) + self.b

    def reg_loss(self):
        return self._l2 * tf.nn.l2_loss(self.w)


class _GreedyTopKPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, scorer: _LinearScorer, slate_size: int, use_affinity: bool = True):
        super().__init__(time_step_spec, action_spec)
        self._scorer = scorer
        self._slate_size = int(slate_size)
        self._use_affinity = bool(use_affinity)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation
        interest   = tf.convert_to_tensor(obs["interest"], tf.float32)       
        item_feats = tf.convert_to_tensor(obs["item_features"], tf.float32)  

        rank = tf.rank(item_feats)
        item_feats_2d = tf.case(
            [(tf.equal(rank, 3), lambda: item_feats[0])],  
            default=lambda: item_feats                     
        )

        # Predicted reward per item
        pred = self._scorer.scores(interest, item_feats_2d)

        score = pred
        if self._use_affinity:
            # Optional: weight by click affinity v(s,i)=interest·item_features[i]
            v_all = tf.linalg.matmul(interest, item_feats_2d, transpose_b=True)
            score = pred * v_all

        top_k = tf.math.top_k(score, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=top_k.indices)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class _EpsGreedyPolicy(tf_policy.TFPolicy):
    def __init__(self, base_policy: _GreedyTopKPolicy, num_items: int, slate_size: int,
                 epsilon=0.2, steps_to_min=20_000, min_epsilon=0.05):
        super().__init__(base_policy.time_step_spec, base_policy.action_spec)
        self._base = base_policy
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)

        self._epsilon = tf.Variable(float(epsilon), trainable=False, dtype=tf.float32)
        decay = (float(min_epsilon) / float(epsilon)) ** (1.0 / float(steps_to_min))
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


# Agent
class CtxBanditAgent(tf_agent.TFAgent):
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=2e-3, l2=1e-6,
                 epsilon=0.2, min_epsilon=0.05, epsilon_decay_steps=20_000,
                 use_affinity_in_policy=True):
        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self.is_learning = True

        dim = int(num_topics)
        self._scorer = _LinearScorer(dim=dim, l2=l2)
        self._opt = tf.keras.optimizers.Adam(learning_rate)
        self._huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)

        # Policies
        greedy = _GreedyTopKPolicy(time_step_spec, action_spec, self._scorer, slate_size, use_affinity=use_affinity_in_policy)
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

    # Replay buffer spec
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

    # training 
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

        # time slices 
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
        reward_SB   = flatten_keep_tail(reward_t, tail_ndims=0)                 

        SB = tf.shape(action_SB)[0]
        slate_size = tf.shape(action_SB)[1]

        clicked_mask = tf.less(click_SB, slate_size)                             # True if real click happened
        safe_click = tf.minimum(click_SB, slate_size - 1)
        idx = tf.stack([tf.range(SB), safe_click], axis=1)                       
        clicked_item_id = tf.gather_nd(action_SB, idx)                           

        with tf.GradientTape() as tape:
            clicked_feats_SB = tf.gather(item_feats_t, clicked_item_id)  

            # Predict clicked reward
            pred_clicked = self._scorer.clicked_pred(interest_SB, clicked_feats_SB) 

            # Huber (or MSE) on immediate reward + L2 regularization
            per_ex = self._huber(reward_SB, pred_clicked)                        
            per_ex = per_ex * tf.cast(clicked_mask, tf.float32)

            denom = tf.reduce_sum(tf.cast(clicked_mask, tf.float32)) + 1e-6
            loss = (tf.reduce_sum(per_ex) / denom) + self._scorer.reg_loss()

        grads = tape.gradient(loss, self._scorer.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self._opt.apply_gradients(zip(grads, self._scorer.trainable_variables))
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
        learning_rate=kw.get("learning_rate", 2e-3),
        l2=kw.get("l2", 1e-6),
        epsilon=kw.get("epsilon", 0.2),
        min_epsilon=kw.get("min_epsilon", 0.05),
        epsilon_decay_steps=kw.get("epsilon_decay_steps", 20_000),
        use_affinity_in_policy=kw.get("use_affinity_in_policy", True),
    )