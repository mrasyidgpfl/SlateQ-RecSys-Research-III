import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

from src.core.registry import register


class _GreedyPolicy:
    """Greedy top‑K by immediate affinity: score = <interest, item_features[i]>."""
    def __init__(self, action_spec):
        self._action_spec = action_spec

    def action(self, time_step: ts.TimeStep):
        obs = time_step.observation

        interest = tf.convert_to_tensor(obs["interest"], dtype=tf.float32)

        item_feats = tf.convert_to_tensor(obs["item_features"], dtype=tf.float32)
        rank = tf.rank(item_feats)
        item_feats = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.tile(tf.expand_dims(item_feats, axis=0), [tf.shape(interest)[0], 1, 1]),
            lambda: item_feats
        )

        scores = tf.einsum("bd,bnd->bn", interest, item_feats)

        # Pick top-K indices per batch row
        slate_size = int(self._action_spec.shape[0])
        topk = tf.math.top_k(scores, k=slate_size)

        max_id = tf.shape(item_feats)[1] - 1
        actions = tf.clip_by_value(topk.indices, 0, max_id)

        return policy_step.PolicyStep(action=actions)

    def decay_epsilon(self, steps=1):
        return


class GreedyAgentShim:
    def __init__(self, time_step_spec, action_spec, **_kwargs):
        self.policy = _GreedyPolicy(action_spec)
        self.collect_policy = self.policy
        self.is_learning = False
        self.collect_data_spec = None


@register("greedy")
def make_greedy(time_step_spec, action_spec, **kwargs):
    return GreedyAgentShim(time_step_spec, action_spec, **kwargs)