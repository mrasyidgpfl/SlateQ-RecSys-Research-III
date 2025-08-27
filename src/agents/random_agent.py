import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from src.core.registry import register

class _RandomPolicy:
    def __init__(self, action_spec):
        self._action_spec = action_spec
        self._is_slate = (len(action_spec.shape) == 1)
        self._slate_size = int(action_spec.shape[0]) if self._is_slate else 1

    def action(self, time_step: ts.TimeStep):
        batch = tf.shape(time_step.step_type)[0]
        minv = tf.cast(self._action_spec.minimum, tf.int32)
        maxv = tf.cast(self._action_spec.maximum, tf.int32)
        num_items = maxv - minv + 1

        if self._is_slate:
            k = tf.minimum(self._slate_size, num_items)
            base = tf.tile(tf.expand_dims(tf.range(minv, maxv + 1, dtype=tf.int32), 0), [batch, 1])
            noise = tf.random.uniform(tf.shape(base), dtype=tf.float32)
            shuffled = tf.gather(base, tf.argsort(noise, axis=1), batch_dims=1)
            slate = shuffled[:, :k]
            def pad_to_k():
                repeat = tf.tile(slate[:, :1], [1, self._slate_size - k])
                return tf.concat([slate, repeat], axis=1)
            action = tf.cond(tf.equal(k, self._slate_size), lambda: slate, pad_to_k)
            action = tf.cast(action, self._action_spec.dtype)
            action = tf.ensure_shape(action, [None, self._slate_size])
        else:
            action = tf.random.uniform([batch], minval=minv, maxval=maxv + 1, dtype=self._action_spec.dtype)
            action = tf.ensure_shape(action, [None])

        return policy_step.PolicyStep(action=action)

    def decay_epsilon(self, steps=1):
        return

class RandomAgentShim:
    def __init__(self, time_step_spec, action_spec, **_kwargs):
        self.policy = _RandomPolicy(action_spec)
        self.collect_policy = self.policy
        self.is_learning = False
        self.collect_data_spec = None

@register("random")
def make_random(time_step_spec, action_spec, **kwargs):
    return RandomAgentShim(time_step_spec, action_spec, **kwargs)
