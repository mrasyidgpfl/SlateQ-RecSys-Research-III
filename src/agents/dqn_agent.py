import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from src.core.registry import register

class _RandomPolicy:
    def __init__(self, action_spec):
        self._action_spec = action_spec

    def action(self, time_step: ts.TimeStep):
        # Batch is [U]; your runtime sets step_type shape=(num_users,)
        batch = tf.shape(time_step.step_type)[0]

        # Generic bounds from spec
        minv = tf.cast(self._action_spec.minimum, tf.int32)  # usually 0
        maxv = tf.cast(self._action_spec.maximum, tf.int32)  # usually num_items-1
        num_items = maxv - minv + 1

        # Slate size from spec shape
        slate_size = int(self._action_spec.shape[0])

        # Sample without replacement: shuffle item ids [minv..maxv], then take first K
        k = tf.minimum(slate_size, num_items)
        # Make a [batch, num_items] matrix of shuffled indices, then slice [:, :k]
        base = tf.tile(tf.expand_dims(tf.range(minv, maxv + 1, dtype=tf.int32), 0), [batch, 1])
        # Add tiny noise row-wise and argsort for a fast shuffle per batch
        noise = tf.random.uniform(tf.shape(base), dtype=tf.float32)
        shuffled = tf.gather(base, tf.argsort(noise, axis=1), batch_dims=1)
        slate = shuffled[:, :k]

        # If someone sets slate_size > num_items, pad with repeats (rare, but safe)
        def pad_to_k():
            # Repeat the first few to match the requested spec shape
            repeat = tf.tile(slate[:, :1], [1, slate_size - k])
            return tf.concat([slate, repeat], axis=1)
        action = tf.cond(tf.equal(k, slate_size), lambda: slate, pad_to_k)

        # Ensure shape matches the spec: [batch, slate_size]
        action = tf.ensure_shape(action, [None, slate_size])
        return policy_step.PolicyStep(action=action)

    def decay_epsilon(self, steps=1):
        # For interface parity with ε-greedy agents
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