import gin
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from recsim_ng.entities.state_models import static
from recsim_ng.core import value

class TensorFieldSpec(value.FieldSpec):
    def __init__(self, shape, dtype=tf.float32):
        super().__init__()
        self._shape = shape
        self._dtype = dtype
    def invariant(self):
        return tf.TensorSpec(shape=self._shape, dtype=self._dtype)

@gin.configurable
class ECommItems(static.StaticStateModel):
    def __init__(self, num_items=100, num_topics=10, init="uniform"):
        super().__init__()
        self.num_items = num_items
        self.num_topics = num_topics
        self.init = init

    def specs(self):
        return value.ValueSpec(
            features=TensorFieldSpec(shape=(self.num_items, self.num_topics), dtype=tf.float32)
        )

    def initial_state(self):
        if self.init == "uniform":
            feats = tf.random.uniform(
                (self.num_items, self.num_topics), minval=-1.0, maxval=1.0, dtype=tf.float32
            )
        else:
            feats = tfd.Normal(loc=0., scale=1.).sample((self.num_items, self.num_topics))
        return value.Value(features=feats)

    def next_state(self, previous_state, *_):
        return previous_state
