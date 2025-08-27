import tensorflow as tf


def dcg_at_k(labels, k):
    labels = tf.cast(labels, tf.float32)
    gains = (2 ** labels - 1)[:, :k]
    discounts = tf.math.log(tf.range(2, k + 2, dtype=tf.float32))
    return tf.reduce_sum(gains / discounts, axis=1)

def ndcg_at_k(true_items, predicted_items, k):
    """
    Computes mean nDCG@k.
    true_items: [batch_size] tensor with ground truth indices (0 ≤ idx < slate_size)
    predicted_items: [batch_size, slate_size] tensor of ranked item indices
    """
    batch_size = tf.shape(predicted_items)[0]
    slate_size = tf.shape(predicted_items)[1]

    # Binary relevance
    relevant = tf.cast(tf.equal(predicted_items, tf.expand_dims(true_items, 1)), tf.float32)

    # DCG
    dcg = dcg_at_k(relevant, k)

    # IDCG: always 1 relevant item ideally at rank 1 → relevance vector is [1, 0, 0, ...]
    ideal_relevance = tf.sort(relevant, direction='DESCENDING')
    idcg = dcg_at_k(ideal_relevance, k)

    ndcg = tf.where(idcg > 0, dcg / idcg, tf.zeros_like(dcg))
    return tf.reduce_mean(ndcg)

def slate_mrr(true_items, predicted_items, k):
    """
    Computes MRR@k.
    true_items: tf.Tensor of shape [batch_size]
    predicted_items: tf.Tensor of shape [batch_size, slate_size]
    k: int
    """
    hits = tf.cast(tf.equal(predicted_items[:, :k], tf.expand_dims(true_items, axis=1)), tf.float32)
    has_hit = tf.reduce_max(hits, axis=1)

    first_hit_pos = tf.argmax(hits, axis=1, output_type=tf.int32)
    reciprocal_ranks = tf.where(
        tf.equal(has_hit, 1.0),
        1.0 / (tf.cast(first_hit_pos, tf.float32) + 1.0),
        tf.zeros_like(first_hit_pos, dtype=tf.float32)
    )

    return tf.reduce_mean(reciprocal_ranks)
