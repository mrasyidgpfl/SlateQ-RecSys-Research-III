import gin
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from recsim_ng.entities.state_models import static
from recsim_ng.core import value as value_lib
from recsim_ng.core.value import ValueSpec, FieldSpec


class TensorFieldSpec(FieldSpec):
    def __init__(self, shape, dtype=tf.float32):
        super().__init__()
        self._shape = shape
        self._dtype = dtype

    def invariant(self):
        return tf.TensorSpec(shape=self._shape, dtype=self._dtype)


@gin.configurable
class ECommUser(static.StaticStateModel):
    """
    Non‑myopic e‑commerce user with stable long‑horizon dynamics
    """

    def __init__(
        self,
        num_topics=10,
        num_users=1,
        beta=5.0,
        reward_mode="sigmoid",

        # Interest dynamics (non‑myopic)
        alpha_learn=0.03,
        beta_fatigue=0.015,
        rho_decay=0.90,
        eta_forget=0.01,
        interest_drift_sigma=0.0,

        # Click‑driven shaping
        alpha_click_pull=0.08,
        very_aligned_cos=0.70,
        adjacent_low_cos=0.30,
        adjacent_high_cos=0.70,
        satiation_mul=0.06,
        novelty_boost_mul=0.04,
        renorm_interests=True,

        # Continuation mix
        b0=-0.90, b1=2.0, b2=1.5, b3=0.2, b4=1.0,

        # Cumulative satisfaction (decayed)
        sat_decay=0.90,
        sat_gain=0.15,
        sat_fatigue=0.10,
        sat_weight=1.0,

        # Diversity bonus
        lambda_div=0.20,

        # Delayed conversions (kept simple + numerically safe)
        c0=-2.0, c1=3.0, c2=1.0, c3=2.5,
        p_delay=0.25,
        price_scale=0.5,
        max_queue=8,
        conv_required_exposures=3.0,
        exposure_decay=0.90,
        gate_steepness=2.0,

        # Cascade + penalties
        pos_weights=(1.0, 0.75, 0.55, 0.40, 0.30),
        no_click_penalty=0.05,
        repeat_penalty=0.15,

        # Training‑safety knobs
        clip_interest=5.0,
        clip_affinity=10.0,
        epsilon_den=1e-6,

        # *** NEVER STOP TRAINING ***
        force_continue=True,
    ):
        super().__init__()
        # Core
        self.num_topics = int(num_topics)
        self.num_users = int(num_users)
        self.beta = float(beta)
        self.reward_mode = str(reward_mode)

        # Dynamics
        self.alpha_learn = float(alpha_learn)
        self.beta_fatigue = float(beta_fatigue)
        self.rho_decay = float(rho_decay)
        self.eta_forget = float(eta_forget)
        self.interest_drift_sigma = float(interest_drift_sigma)

        # Click shaping
        self.alpha_click_pull = float(alpha_click_pull)
        self.very_aligned_cos = float(very_aligned_cos)
        self.adjacent_low_cos = float(adjacent_low_cos)
        self.adjacent_high_cos = float(adjacent_high_cos)
        self.satiation_mul = float(satiation_mul)
        self.novelty_boost_mul = float(novelty_boost_mul)
        self.renorm_interests = bool(renorm_interests)

        # Continuation
        self.b0, self.b1, self.b2, self.b3, self.b4 = map(float, (b0, b1, b2, b3, b4))

        # Satisfaction
        self.sat_decay = float(sat_decay)
        self.sat_gain = float(sat_gain)
        self.sat_fatigue = float(sat_fatigue)
        self.sat_weight = float(sat_weight)

        # Reward shaping
        self.lambda_div = float(lambda_div)

        # Conversions + gate
        self.c0, self.c1, self.c2, self.c3 = map(float, (c0, c1, c2, c3))
        self.p_delay = float(p_delay)
        self.price_scale = float(price_scale)
        self.max_queue = int(max_queue)
        self.conv_required_exposures = float(conv_required_exposures)
        self.exposure_decay = float(exposure_decay)
        self.gate_steepness = float(gate_steepness)

        # Cascade
        self.pos_weights = tf.constant(pos_weights, dtype=tf.float32)
        self.no_click_penalty = float(no_click_penalty)
        self.repeat_penalty = float(repeat_penalty)

        # Safety
        self.clip_interest = float(clip_interest)
        self.clip_affinity = float(clip_affinity)
        self.epsilon_den = float(epsilon_den)

        # Never stop
        self.force_continue = bool(force_continue)

    # Specs / init
    def specs(self):
        return ValueSpec(
            # persistent user state
            interest=TensorFieldSpec((self.num_users, self.num_topics), tf.float32),
            recent_count=TensorFieldSpec((self.num_users, self.num_topics), tf.float32),
            exposure_streak=TensorFieldSpec((self.num_users, self.num_topics), tf.float32),
            novelty_prev=TensorFieldSpec((self.num_users,), tf.float32),
            novelty_momentum=TensorFieldSpec((self.num_users,), tf.float32),
            satisfaction_logit=TensorFieldSpec((self.num_users,), tf.float32),
            q_k=TensorFieldSpec((self.num_users, self.max_queue), tf.int32),
            q_v=TensorFieldSpec((self.num_users, self.max_queue), tf.float32),

            # last step outputs (for logging + runtime)
            choice=TensorFieldSpec((self.num_users,), tf.int32),
            reward=TensorFieldSpec((self.num_users,), tf.float32),
            continue_flag=TensorFieldSpec((self.num_users,), tf.int32),
        )

    def initial_state(self):
        # Interests start standard normal
        interest = tfd.Normal(0., 1.).sample((self.num_users, self.num_topics))
        interest = tf.clip_by_value(tf.cast(interest, tf.float32),
                                    -self.clip_interest, self.clip_interest)

        zeros_ut = tf.zeros([self.num_users, self.num_topics], tf.float32)
        zeros_u = tf.zeros([self.num_users], tf.float32)
        q_k = tf.fill([self.num_users, self.max_queue], tf.cast(-1, tf.int32))
        q_v = tf.zeros([self.num_users, self.max_queue], tf.float32)

        return value_lib.Value(
            interest=interest,
            recent_count=zeros_ut,
            exposure_streak=zeros_ut,
            novelty_prev=zeros_u,
            novelty_momentum=zeros_u,
            satisfaction_logit=zeros_u,
            q_k=q_k, q_v=q_v,
            choice=tf.zeros([self.num_users], tf.int32),
            reward=tf.zeros([self.num_users], tf.float32),
            continue_flag=tf.ones([self.num_users], tf.int32),
        )

    # Helpers
    def _pairwise_diversity(self, slate_feats):
        K = tf.shape(slate_feats)[1]
        no_pairs = K < 2
        nf = tf.nn.l2_normalize(tf.cast(slate_feats, tf.float32), axis=2)
        sims = tf.matmul(nf, nf, transpose_b=True)
        sum_all = tf.reduce_sum(sims, axis=[1, 2])
        sum_diag = tf.reduce_sum(tf.linalg.diag_part(sims), axis=1)
        sum_off = sum_all - sum_diag
        count_off = tf.cast(K * (K - 1), tf.float32)
        mean_off = tf.math.divide_no_nan(sum_off, tf.maximum(count_off, 1.0))
        div = 1.0 - mean_off
        return tf.where(no_pairs, tf.zeros_like(div), div)

    def _item_topic_weights(self, item_feats):
        # Stable softmax over topics
        return tf.nn.softmax(tf.cast(item_feats, tf.float32), axis=-1)

    def _novelty(self, recent_count, slate_topic_w):
        expo = tf.einsum('ut,ukt->uk', tf.cast(recent_count, tf.float32), slate_topic_w)
        novelty_item = 1.0 / (1.0 + tf.maximum(expo, 0.0))
        return tf.reduce_mean(novelty_item, axis=1), novelty_item

    def _queue_mature_value(self, q_k, q_v):
        matur_mask = tf.equal(q_k, 1)
        return tf.reduce_sum(tf.where(matur_mask, q_v, 0.0), axis=1)

    def _truncate_or_pad_weights(self, K):
        w = self.pos_weights
        w_len = tf.shape(w)[0]

        def pad():
            pad_len = K - w_len
            return tf.concat([w, tf.fill([pad_len], w[-1])], axis=0)

        def trunc():
            return w[:K]

        return tf.cond(w_len < K, pad, trunc)

    # Step functions
    def response(self, user_state, slate, item_state):
        """
        Compute choice, reward, continuation.
        choice ∈ [0..K]; K means "no click".
        """
        slate_idx = slate.get('slate')
        feats = item_state.get('features')
        gathered = tf.gather(feats, slate_idx)

        interest = user_state.get('interest')
        # Affinities (clipped for stability)
        affinities = tf.einsum('ut,ukt->uk',
                               tf.cast(interest, tf.float32),
                               tf.cast(gathered, tf.float32))
        affinities = tf.clip_by_value(affinities, -self.clip_affinity, self.clip_affinity)

        # Cascading click with position bias
        K = tf.shape(slate_idx)[1]
        w = self._truncate_or_pad_weights(K)
        p = tf.nn.sigmoid(self.beta * affinities) * w
        p = tf.clip_by_value(p, 0.0, 1.0 - 1e-6)
        surv = tf.math.cumprod(1.0 - p, axis=1, exclusive=True)
        s = p * surv
        p_no = tf.reduce_prod(1.0 - p, axis=1, keepdims=True)
        probs = tf.concat([s, p_no], axis=1)
        probs = tf.math.divide_no_nan(
            probs, tf.reduce_sum(probs, axis=1, keepdims=True) + self.epsilon_den
        )

        # Sample choice
        choice = tf.cast(tfd.Categorical(probs=probs).sample(), tf.int32)
        clicked = choice < K
        safe_choice = tf.minimum(choice, K - 1)

        # Base reward (safe gather)
        chosen_aff = tf.gather(affinities, safe_choice[:, None], batch_dims=1)[:, 0]
        if self.reward_mode == "sigmoid":
            base_clicked = tf.nn.sigmoid(chosen_aff)
        elif self.reward_mode == "clip01":
            a_min = tf.reduce_min(affinities, axis=1, keepdims=True)
            a_max = tf.reduce_max(affinities, axis=1, keepdims=True)
            norm = (affinities - a_min) / tf.maximum(a_max - a_min, self.epsilon_den)
            base_clicked = tf.gather(norm, safe_choice[:, None], batch_dims=1)[:, 0]
        else:
            base_clicked = tf.cast(chosen_aff, tf.float32)

        base_reward = tf.where(
            clicked,
            base_clicked,
            -tf.fill(tf.shape(base_clicked), tf.constant(self.no_click_penalty, tf.float32))
        )

        # Diversity bonus
        div = self._pairwise_diversity(gathered)

        # Repeat exposure penalty
        topic_w = self._item_topic_weights(feats)
        slate_topic_w = tf.gather(topic_w, slate_idx)
        rep_signal = tf.einsum('ut,ukt->u', user_state.get('recent_count'), slate_topic_w)
        rep_signal = rep_signal / tf.maximum(tf.cast(K, tf.float32), 1.0)
        rep_pen = self.repeat_penalty * rep_signal

        # Conversions payout
        matured = self._queue_mature_value(user_state.get('q_k'), user_state.get('q_v'))

        reward = tf.cast(base_reward + self.lambda_div * div + matured - rep_pen, tf.float32)

        # Continuation
        novelty_mean, _ = self._novelty(user_state.get('recent_count'), slate_topic_w)
        avg_aff = tf.reduce_mean(affinities, axis=1)
        cont_logits = (
            self.b0
            + self.b1 * div
            + self.b2 * novelty_mean
            + self.b4 * user_state.get('novelty_momentum')
            + self.b3 * avg_aff
            + self.sat_weight * user_state.get('satisfaction_logit')
        )
        if self.force_continue:
            continue_flag = tf.ones([self.num_users], tf.int32)
        else:
            continue_flag = tf.cast(tfd.Bernoulli(logits=cont_logits).sample(), tf.int32)

        return value_lib.Value(choice=choice, reward=reward, continue_flag=continue_flag)

    def next_state(self, user_state, slate, item_state, response=None):
        """
        Apply transitions after response (non‑myopic):
          - interest/recent_count/exposure_streak updates
          - click‑driven shaping (pull + norm shaping)
          - conversions queue tick/push (spaced‑exposure gate)
          - novelty momentum
          - cumulative satisfaction logit
        """
        interest = user_state.get('interest')
        recent = user_state.get('recent_count')
        streak = user_state.get('exposure_streak')
        nov_prev = user_state.get('novelty_prev')
        nov_mom = user_state.get('novelty_momentum')
        sat_prev = user_state.get('satisfaction_logit')
        q_k = user_state.get('q_k')
        q_v = user_state.get('q_v')

        slate_idx = slate.get('slate')
        feats = item_state.get('features')
        gathered = tf.gather(feats, slate_idx)
        topic_w_all = self._item_topic_weights(feats)
        slate_topic_w = tf.gather(topic_w_all, slate_idx)

        # Exposure aggregates per topic
        exposure = tf.reduce_sum(slate_topic_w, axis=1)

        # Base interest evolution
        recent_next = self.rho_decay * recent + exposure
        interest_next = (
            (1.0 - self.eta_forget) * interest
            + self.alpha_learn * exposure
            - self.beta_fatigue * recent_next
        )

        # Optional small drift
        if self.interest_drift_sigma > 0.0:
            noise = tfd.Normal(0., self.interest_drift_sigma).sample(tf.shape(interest_next))
            interest_next = interest_next + tf.cast(noise, tf.float32)

        # Clamp + optional renorm for safety
        interest_next = tf.clip_by_value(interest_next, -self.clip_interest, self.clip_interest)
        if self.renorm_interests:
            interest_next = tf.nn.l2_normalize(interest_next, axis=-1)

        # Tick queue and clear matured
        k_dec = tf.where(q_k > 0, q_k - 1, q_k)
        matured_mask = tf.equal(k_dec, 0)
        q_k_cleared = tf.where(matured_mask, tf.fill(tf.shape(k_dec), -1), k_dec)
        q_v_cleared = tf.where(matured_mask, tf.zeros_like(q_v), q_v)

        # Compute choice signals safely
        K = tf.shape(slate_idx)[1]
        if response is not None:
            choice = tf.cast(response.get('choice'), tf.int32)
        else:
            aff_now = tf.einsum('ut,ukt->uk', interest, gathered)
            choice = tf.cast(tf.argmax(aff_now, axis=1), tf.int32)

        clicked = choice < K
        safe_choice = tf.minimum(choice, K - 1)

        # Signals on chosen item
        aff = tf.einsum('ut,ukt->uk', interest, gathered)
        aff = tf.clip_by_value(aff, -self.clip_affinity, self.clip_affinity)
        chosen_aff = tf.gather(aff, safe_choice[:, None], batch_dims=1)[:, 0]
        nov_mean, novelty_item = self._novelty(recent, slate_topic_w)
        chosen_nov = tf.gather(novelty_item, safe_choice[:, None], batch_dims=1)[:, 0]
        chosen_w = tf.gather(slate_topic_w, safe_choice[:, None], batch_dims=1)[:, 0, :]

        # Click‑driven shaping
        interest_n = tf.nn.l2_normalize(interest, axis=-1)
        chosen_n = tf.nn.l2_normalize(chosen_w, axis=-1)
        cos = tf.reduce_sum(interest_n * chosen_n, axis=-1)

        very_aligned = tf.cast(cos > self.very_aligned_cos, tf.float32)
        adjacent = tf.cast((cos > self.adjacent_low_cos) & (cos <= self.adjacent_high_cos), tf.float32)
        clicked_f = tf.cast(clicked, tf.float32)[:, None]

        # Pull toward chosen topic weights
        interest_next = interest_next + self.alpha_click_pull * clicked_f * chosen_w

        # Norm shaping on click
        scale = (1.0 - self.satiation_mul * very_aligned[:, None]) + (self.novelty_boost_mul * adjacent[:, None])
        interest_next = tf.where(clicked_f > 0.0, interest_next * scale, interest_next)

        # Safety again after shaping
        interest_next = tf.clip_by_value(interest_next, -self.clip_interest, self.clip_interest)
        if self.renorm_interests:
            interest_next = tf.nn.l2_normalize(interest_next, axis=-1)

        # Exposure streak + spaced exposure gate
        streak_next = self.exposure_decay * streak + exposure
        eff_expo = tf.einsum('ut,ut->u', streak_next, chosen_w)
        gate = tf.nn.sigmoid(self.gate_steepness * (eff_expo - self.conv_required_exposures))

        # Conversions (only if clicked) — numerically safe
        conv_logits = self.c0 + self.c1 * chosen_aff + self.c2 * chosen_nov + self.c3 * gate
        conv_sample = tf.cast(tfd.Bernoulli(logits=conv_logits).sample(), tf.bool)
        conv = tf.logical_and(clicked, conv_sample)

        delay = tf.cast(tfd.Geometric(probs=tf.clip_by_value(self.p_delay, 1e-3, 1.0)).sample([self.num_users]),
                        tf.int32) + 1
        delay = tf.minimum(delay, tf.fill([self.num_users], tf.cast(10, tf.int32)))
        val = self.price_scale * tf.cast(tf.nn.relu(chosen_aff), tf.float32)

        empty = tf.equal(q_k_cleared, -1)
        any_empty = tf.reduce_any(empty, axis=1)
        pos = tf.argmax(tf.cast(empty, tf.int32), axis=1, output_type=tf.int32)
        idx = tf.stack([tf.range(self.num_users, dtype=tf.int32), pos], axis=1)
        pushable = tf.logical_and(conv, any_empty)

        k_at = tf.gather_nd(q_k_cleared, idx)
        v_at = tf.gather_nd(q_v_cleared, idx)
        k_upd = tf.where(pushable, delay, k_at)
        v_upd = tf.where(pushable, val,   v_at)
        q_k_next = tf.tensor_scatter_nd_update(q_k_cleared, idx, k_upd)
        q_v_next = tf.tensor_scatter_nd_update(q_v_cleared, idx, v_upd)

        # Novelty momentum
        mom_next = 0.8 * nov_mom + 0.2 * (nov_mean - nov_prev)
        nov_prev_next = nov_mean

        # Satisfaction signal (only when clicked)
        base_click_signal = tf.nn.sigmoid(chosen_aff)
        base_click_signal = tf.where(clicked, base_click_signal, tf.zeros_like(base_click_signal))
        fatigue_pen = self.sat_fatigue * tf.cast(cos > self.very_aligned_cos, tf.float32)
        sat_next = self.sat_decay * sat_prev + self.sat_gain * base_click_signal - fatigue_pen

        # Carry current step outputs for logging consistency
        if response is not None:
            choice_cur = tf.cast(response.get('choice'), tf.int32)
            reward_cur = tf.cast(response.get('reward'), tf.float32)
            if self.force_continue:
                cont_cur = tf.ones_like(choice_cur, dtype=tf.int32)
            else:
                cont_cur = tf.cast(response.get('continue_flag'), tf.int32)
        else:
            # Fallbacks
            choice_cur = tf.cast(choice, tf.int32)
            reward_cur = tf.zeros_like(choice_cur, dtype=tf.float32)
            cont_cur = tf.ones_like(choice_cur, dtype=tf.int32) if self.force_continue else tf.ones_like(choice_cur, dtype=tf.int32)

        return value_lib.Value(
            interest=interest_next,
            recent_count=recent_next,
            exposure_streak=streak_next,
            novelty_prev=nov_prev_next,
            novelty_momentum=mom_next,
            satisfaction_logit=sat_next,
            q_k=q_k_next, q_v=q_v_next,

            # carry current step outputs for introspection
            choice=choice_cur,
            reward=reward_cur,
            continue_flag=cont_cur,
        )
