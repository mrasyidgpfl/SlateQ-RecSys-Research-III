from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec
from recsim_ng.core import value as value_lib


class ECommRuntime(runtime.TFRuntime):
    def __init__(self, network):
        super().__init__(network)
        self._current_state = self._network.initial_step()

        user_state = self._current_state.get("user_state", {})
        item_state = self._current_state.get("item_state", {})
        rec_state  = self._current_state.get("rec_state", {})

        interest   = user_state.get("interest")
        item_feats = item_state.get("features")
        slate0     = rec_state.get("slate")

        self._num_users    = int(interest.shape[0]) if interest is not None else 10
        self._interest_dim = int(interest.shape[1]) if (interest is not None and len(interest.shape) > 1) else 10

        if item_feats is not None and len(item_feats.shape) >= 2:
            self._num_items  = int(item_feats.shape[0])
            self._num_topics = int(item_feats.shape[1])
        else:
            self._num_items  = 100
            self._num_topics = self._interest_dim

        if slate0 is not None and len(slate0.shape) >= 2:
            self._slate_size = int(slate0.shape[1])
        else:
            self._slate_size = 5

        self._user_range = tf.range(self._num_users, dtype=tf.int32)

        self._action_spec = tensor_spec.BoundedTensorSpec(
            shape=(self._slate_size,),
            dtype=tf.int32,
            minimum=0,
            maximum=self._num_items - 1,
            name="slate_indices",
        )

    def reset(self):
        self._current_state = self._network.initial_step()
        return self._to_first_time_step(self._current_state)

    def step(self, action):
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        action = tf.clip_by_value(action, 0, self._num_items - 1)

        inputs = {
            "action":      value_lib.Value(act=action),
            "response":    self._current_state["response"],
            "user_state":  self._current_state["user_state"],
            "rec_state":   self._current_state["rec_state"],
            "item_state":  self._current_state["item_state"],
        }
        self._current_state = self._network.step(inputs)
        return self._to_mid_or_last_time_step(self._current_state)

    def observation_spec(self):
        return {
            "interest": tf.TensorSpec(shape=(self._num_users, self._interest_dim), dtype=tf.float32),
            "choice":   tf.TensorSpec(shape=(self._num_users,), dtype=tf.int32),
            "item_features": tf.TensorSpec(
                shape=(self._num_users, self._num_items, self._num_topics), dtype=tf.float32
            ),
        }

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return ts.TimeStep(
            step_type=tf.TensorSpec(shape=(self._num_users,), dtype=tf.int32),
            reward=tf.TensorSpec(shape=(self._num_users,), dtype=tf.float32),
            discount=tf.TensorSpec(shape=(self._num_users,), dtype=tf.float32),
            observation=self.observation_spec(),
        )

    def trajectory(self, num_steps=5000):
        self._current_state = self._network.initial_step()
        packed_state = self._pack(self._current_state)

        def cond(step, *_):
            return tf.less(step, num_steps)

        def body(step, state):
            next_state = self._packed_step(state)
            return step + 1, next_state

        shape_invariants = [tf.TensorShape([])] + [
            tf.TensorShape(p.shape) if isinstance(p, tf.Tensor) else tf.TensorShape([])
            for p in packed_state
        ]

        _, final_state = tf.while_loop(
            cond, body, [0, packed_state], shape_invariants=shape_invariants
        )
        return self._unpack(final_state)

    def _to_first_time_step(self, state_value):
        user_state = state_value.get("user_state")
        item_state = state_value.get("item_state")

        interest   = user_state.get("interest")
        item_feats = item_state.get("features")
        num_users  = tf.shape(interest)[0]

        item_feats_batched = tf.tile(item_feats[None, ...], [num_users, 1, 1])
        choice0 = tf.zeros([num_users], dtype=tf.int32)

        return ts.TimeStep(
            step_type=tf.fill([num_users], ts.StepType.FIRST),
            reward=tf.zeros([num_users], tf.float32),
            discount=tf.ones([num_users], tf.float32),
            observation={
                "interest":      interest,
                "choice":        choice0,
                "item_features": item_feats_batched,
            },
        )

    def _to_mid_or_last_time_step(self, state_value):
        user_state = state_value.get("user_state")
        response   = state_value.get("response")
        item_state = state_value.get("item_state")

        interest   = user_state.get("interest")
        reward     = response.get("reward")
        choice     = response.get("choice")
        cont_flag  = response.get("continue_flag")
        item_feats = item_state.get("features")

        num_users = tf.shape(interest)[0]
        item_feats_batched = tf.tile(item_feats[None, ...], [num_users, 1, 1])

        is_last   = tf.equal(cont_flag, 0)
        step_type = tf.where(
            is_last,
            tf.fill(tf.shape(cont_flag), ts.StepType.LAST),
            tf.fill(tf.shape(cont_flag), ts.StepType.MID)
        )
        discount = tf.where(is_last, tf.zeros_like(reward), tf.ones_like(reward))

        return ts.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation={
                "interest":      interest,
                "choice":        choice,
                "item_features": item_feats_batched,
            },
        )

    def alive_mask(self):
        cont_flag = self._current_state.get("response").get("continue_flag")
        return tf.equal(cont_flag, 1)
