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
    """
    Mixed-entropy catalog with mild churn:
      - A fraction of items are 'clickbait' (near one-hot topic mass).
      - The rest are 'slow-burn' (higher-entropy, multi-topic).
      - Each tick, a small random subset is refreshed (churn), keeping proportions.
    Only outputs `features` to stay backward compatible.
    """
    def __init__(
        self,
        num_items=100,
        num_topics=10,
        # mixture
        frac_clickbait=0.6,
        # Dirichlet-style generators (must be >0)
        clickbait_conc=0.05,
        clickbait_spike=25.0,
        slowburn_conc=2.0,
        # shaping
        spread_scale=3.0,
        noise_std=0.05,
        l2_normalize=True,
        # catalog churn
        churn_prob=0.02,
        seed=None,
    ):
        super().__init__()
        self.num_items = int(num_items)
        self.num_topics = int(num_topics)
        self.frac_clickbait = float(frac_clickbait)

        self.clickbait_conc = float(clickbait_conc)
        self.clickbait_spike = float(clickbait_spike)
        self.slowburn_conc = float(slowburn_conc)

        self.spread_scale = float(spread_scale)
        self.noise_std = float(noise_std)
        self.l2_normalize = bool(l2_normalize)

        self.churn_prob = float(churn_prob)
        self._seed = seed

    def specs(self):
        return value.ValueSpec(
            features=TensorFieldSpec(shape=(self.num_items, self.num_topics), dtype=tf.float32)
        )

    def _sample_clickbait(self, n):
        if n <= 0:
            return tf.zeros([0, self.num_topics], tf.float32)
        rng = None if self._seed is None else tf.random.Generator.from_seed(self._seed).make_seeds(1)[0]
        # choose a dominant topic per item
        dom_idx = tfd.Categorical(probs=tf.ones([self.num_topics]) / self.num_topics).sample(n, seed=rng)
        dom_onehot = tf.one_hot(dom_idx, depth=self.num_topics, dtype=tf.float32)
        base = tf.fill([n, self.num_topics], self.clickbait_conc)
        alpha = base + self.clickbait_spike * dom_onehot
        w = tfd.Dirichlet(alpha).sample()
        f = (w - (1.0 / self.num_topics)) * self.spread_scale
        if self.noise_std > 0.0:
            f = f + tf.random.normal(tf.shape(f), stddev=self.noise_std, dtype=tf.float32, seed=rng)
        if self.l2_normalize:
            f = tf.math.l2_normalize(f, axis=-1)
        return tf.cast(f, tf.float32)

    def _sample_slowburn(self, n):
        if n <= 0:
            return tf.zeros([0, self.num_topics], tf.float32)
        rng = None if self._seed is None else tf.random.Generator.from_seed(self._seed).make_seeds(1)[0]
        alpha = tf.fill([self.num_topics], self.slowburn_conc)
        w = tfd.Dirichlet(alpha).sample(n, seed=rng)
        f = (w - (1.0 / self.num_topics)) * self.spread_scale
        if self.noise_std > 0.0:
            f = f + tf.random.normal(tf.shape(f), stddev=self.noise_std, dtype=tf.float32, seed=rng)
        if self.l2_normalize:
            f = tf.math.l2_normalize(f, axis=-1)
        return tf.cast(f, tf.float32)

    def _make_catalog(self, n_items):
        n_click = tf.cast(tf.round(self.frac_clickbait * n_items), tf.int32)
        n_slow = tf.cast(n_items - n_click, tf.int32)
        cb = self._sample_clickbait(int(n_click))
        sb = self._sample_slowburn(int(n_slow))
        feats = tf.concat([cb, sb], axis=0)
        # shuffle to avoid any ordering bias
        perm = tf.random.shuffle(tf.range(tf.shape(feats)[0], dtype=tf.int32))
        feats = tf.gather(feats, perm)
        return feats

    def initial_state(self):
        feats = self._make_catalog(self.num_items)
        return value.Value(features=tf.cast(feats, tf.float32))

    def next_state(self, previous_state, *_):
        if self.churn_prob <= 0.0:
            return previous_state
        feats = previous_state.get('features')
        n = tf.shape(feats)[0]
        # decide which items to churn
        churn_mask = tfd.Bernoulli(probs=self.churn_prob).sample(sample_shape=[n], dtype=tf.bool)
        num_churn = tf.reduce_sum(tf.cast(churn_mask, tf.int32))
        # resample replacements using the same mixture process
        new_feats = self._make_catalog(tf.cast(num_churn, tf.int32))
        # scatter-update
        idx = tf.where(churn_mask)[:, 0]  # [num_churn]
        feats_updated = tf.tensor_scatter_nd_update(feats, tf.expand_dims(idx, 1), new_feats)
        return value.Value(features=tf.cast(feats_updated, tf.float32))
