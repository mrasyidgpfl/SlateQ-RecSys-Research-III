import os
import sys
from contextlib import contextmanager
import logging
import absl.logging
import warnings
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLBACKEND', 'Agg')

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)
warnings.filterwarnings('ignore', message='Index out of range using input dim')

@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

import numpy as np
import tensorflow as tf
np.random.seed(1337)
tf.random.set_seed(1337)
try:
    import multiprocessing as mp
    perf = max(2, mp.cpu_count() // 2)
    tf.config.threading.set_intra_op_parallelism_threads(perf)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

tf.get_logger().setLevel(logging.FATAL)
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.debugging.disable_check_numerics()
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

with suppress_stderr():
    import matplotlib
    matplotlib.use(os.environ['MPLBACKEND'])
    import matplotlib.pyplot as plt

import pandas as pd
import gin
from absl import app
from absl import flags
from pathlib import Path

from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.trajectories import time_step as ts

from src.runtimes.ecomm_runtime import ECommRuntime
from src.stories import ecomm_story
from src.core.registry import REGISTRY, register

import src.agents.random_agent
import src.agents.greedy_agent
import src.agents.ctxbandit_agent
from src.agents.dqn_agent import DQNAgent
from src.agents.slateq_agent import SlateQAgent
from src.agents.slateq_dueling_agent import SlateQDuelingAgent
from src.agents.slateq_noisynet_agent import SlateQNoisyNetAgent

from src.metrics.ranking_metrics import ndcg_at_k, slate_mrr
from src.metrics.logger import MetricsLogger

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Paths to config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin parameter bindings.')
flags.DEFINE_string('agent', 'slateq',
                    'Agent: random | greedy | ctxbandit | slateq | slateq_noisynet | dqn')

flags.DEFINE_integer('episodes', 600, 'Number of episodes.')
flags.DEFINE_integer('steps', 200, 'Steps per episode.')
flags.DEFINE_integer('num_users', 10, 'Users per batch.')
flags.DEFINE_integer('num_items', 100, 'Catalog size.')
flags.DEFINE_integer('slate_size', 5, 'Slate size K.')

flags.DEFINE_integer('warmup_frames', 5000, 'Frames before training starts.')
flags.DEFINE_integer('replay_capacity', 10000, 'Replay buffer capacity.')
flags.DEFINE_integer('batch_size', 32, 'Replay sample batch size.')

flags.DEFINE_bool('plot', True, 'Generate plots at end.')
flags.DEFINE_bool('peek_final_state', False, 'Print final user interest (can be heavy).')

@register("slateq")
def make_slateq(time_step_spec, action_spec, **kwargs):
    return SlateQAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        num_users=kwargs.get("num_users"),
        num_topics=kwargs.get("num_topics", 10),
        slate_size=kwargs.get("slate_size"),
        num_items=kwargs.get("num_items"),
        learning_rate=kwargs.get("learning_rate", 1e-3),
        epsilon=kwargs.get("epsilon", 0.2),
        min_epsilon=kwargs.get("min_epsilon", 0.05),
        epsilon_decay_steps=kwargs.get("epsilon_decay_steps", 20000),
        target_update_period=kwargs.get("target_update_period", 1000),
        gamma=kwargs.get("gamma", 0.95),
        beta=kwargs.get("beta", 5.0),
    )

@register("slateqdueling")
def make_slateq_dueling(time_step_spec, action_spec, **kwargs):
    return SlateQDuelingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        num_users=kwargs.get("num_users"),
        num_topics=kwargs.get("num_topics", 10),
        slate_size=kwargs.get("slate_size"),
        num_items=kwargs.get("num_items"),
        learning_rate=kwargs.get("learning_rate", 1e-3),
        epsilon=kwargs.get("epsilon", 0.2),
        min_epsilon=kwargs.get("min_epsilon", 0.05),
        epsilon_decay_steps=kwargs.get("epsilon_decay_steps", 20000),
        target_update_period=kwargs.get("target_update_period", 1000),
        gamma=kwargs.get("gamma", 0.95),
        beta=kwargs.get("beta", 5.0),
    )

@register("slateq_noisynet")
def make_slateq_noisynet(time_step_spec, action_spec, **kwargs):
    return SlateQNoisyNetAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        num_users=kwargs.get("num_users"),
        num_topics=kwargs.get("num_topics", 10),
        slate_size=kwargs.get("slate_size"),
        num_items=kwargs.get("num_items"),
        learning_rate=kwargs.get("learning_rate", 1e-3),
        epsilon=kwargs.get("epsilon", 0.1),
        min_epsilon=kwargs.get("min_epsilon", 0.05),
        epsilon_decay_steps=kwargs.get("epsilon_decay_steps", int(0.6 * kwargs.get("episodes", 600) * kwargs.get("steps", 200))),
        target_update_period=kwargs.get("target_update_period", 1000),
        tau=kwargs.get("tau", 0.005),
        gamma=kwargs.get("gamma", 0.95),
        beta=kwargs.get("beta", 5.0),
        huber_delta=kwargs.get("huber_delta", 1.0),
        grad_clip_norm=kwargs.get("grad_clip_norm", 10.0),
        reward_scale=kwargs.get("reward_scale", 10.0),
        l2=kwargs.get("l2", 0.0),
        pos_weights=kwargs.get("pos_weights", None),
        noisy_std_init=kwargs.get("noisy_std_init", 0.5),
        reset_noise_every=kwargs.get("reset_noise_every", 1),
    )

@register("dqn")
def make_dqn(time_step_spec, action_spec, **kwargs):
    return DQNAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        num_users=kwargs.get("num_users"),
        num_topics=kwargs.get("num_topics", 10),
        slate_size=kwargs.get("slate_size"),
        num_items=kwargs.get("num_items"),
        learning_rate=kwargs.get("learning_rate", 1e-3),
        epsilon=kwargs.get("epsilon", 0.2),
        min_epsilon=kwargs.get("min_epsilon", 0.05),
        epsilon_decay_steps=kwargs.get("epsilon_decay_steps", int(0.6 * kwargs.get("episodes", 600) * kwargs.get("steps", 200))),
        target_update_period=kwargs.get("target_update_period", 1000),
        tau=kwargs.get("tau", 0.005),
        gamma=kwargs.get("gamma", 0.95),
        huber_delta=kwargs.get("huber_delta", 1.0),
        grad_clip_norm=kwargs.get("grad_clip_norm", 10.0),
        reward_scale=kwargs.get("reward_scale", 10.0),
        l2=kwargs.get("l2", 0.0),
    )

def _make_agent_timestep_spec(num_items, num_topics):
    return ts.TimeStep(
        step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
        reward=tf.TensorSpec(shape=(), dtype=tf.float32),
        discount=tf.TensorSpec(shape=(), dtype=tf.float32),
        observation={
            "interest": tf.TensorSpec(shape=(num_topics,), dtype=tf.float32),
            "choice": tf.TensorSpec(shape=(), dtype=tf.int32),
            "item_features": tf.TensorSpec(shape=(num_items, num_topics), dtype=tf.float32),
        },
    )

def _ensure_batched_item_feats(timestep: ts.TimeStep, num_users: int):
    obs = dict(timestep.observation)
    feats = obs["item_features"]
    if tf.rank(feats) == 2:
        feats = tf.tile(feats[None, ...], [num_users, 1, 1])
    obs["item_features"] = feats
    return timestep._replace(observation=obs)

def main(argv):
    pos_agent = None
    if len(argv) > 1 and not argv[1].startswith('-'):
        pos_agent = argv[1]
        argv = [argv[0]] + [a for a in argv[1:] if a.startswith('-')]
    FLAGS(argv)
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

    agent_name = pos_agent if pos_agent is not None else FLAGS.agent
    if agent_name not in REGISTRY:
        agent_name = "random"

    num_users = FLAGS.num_users
    slate_size = FLAGS.slate_size
    num_items = FLAGS.num_items
    num_episodes = FLAGS.episodes
    steps_per_episode = FLAGS.steps
    warmup_frames = FLAGS.warmup_frames
    replay_capacity = FLAGS.replay_capacity
    batch_size = FLAGS.batch_size
    num_topics = 10

    network = ecomm_story(num_users=num_users, num_items=num_items, slate_size=slate_size)
    rt = ECommRuntime(network=network)

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(slate_size,), dtype=tf.int32, minimum=0, maximum=num_items - 1
    )
    agent_time_step_spec = _make_agent_timestep_spec(num_items, num_topics)

    agent = REGISTRY[agent_name](
        time_step_spec=agent_time_step_spec,
        action_spec=action_spec,
        num_users=num_users,
        slate_size=slate_size,
        num_items=num_items,
        num_topics=num_topics,
        epsilon_decay_steps=int(0.6 * num_episodes * steps_per_episode),
    )

    if getattr(agent, "is_learning", False) and hasattr(agent.collect_policy, "epsilon"):
        try:
            print(f"Init ε = {float(agent.collect_policy.epsilon):.3f}")
        except Exception:
            pass

    dummy_ts = ts.TimeStep(
        step_type=tf.zeros((num_users,), tf.int32),
        reward=tf.zeros((num_users,), tf.float32),
        discount=tf.ones((num_users,), tf.float32),
        observation={
            "interest": tf.zeros((num_users, num_topics), tf.float32),
            "choice": tf.zeros((num_users,), tf.int32),
            "item_features": tf.zeros((num_users, num_items, num_topics), tf.float32),
        },
    )
    _ = agent.policy.action(dummy_ts)

    dataset = None
    dataset_iter = None
    is_learning = bool(getattr(agent, "is_learning", False))
    if is_learning:
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=num_users,
            max_length=replay_capacity,
        )
        dataset = (replay_buffer
            .as_dataset(
                num_parallel_calls=tf.data.AUTOTUNE,
                sample_batch_size=batch_size,
                num_steps=2,
                single_deterministic_pass=False
            )
            .prefetch(tf.data.AUTOTUNE))
        dataset_iter = iter(dataset)

    logger = MetricsLogger(base_dir="logs/" + agent_name)

    try:
        for episode in range(num_episodes):
            episode_losses = []
            time_step = rt.reset()
            time_step = _ensure_batched_item_feats(time_step, num_users)

            episode_reward_t = tf.constant(0.0, dtype=tf.float32)
            last_slate = None
            last_choice = None
            last_reward = None

            for step in range(steps_per_episode):
                action_step = agent.collect_policy.action(time_step)
                action = tf.clip_by_value(action_step.action, 0, num_items - 1)

                next_time_step = rt.step(action)
                next_time_step = _ensure_batched_item_feats(next_time_step, num_users)

                last_slate = action
                last_reward = next_time_step.reward
                last_choice = next_time_step.observation["choice"]
                episode_reward_t += tf.reduce_sum(last_reward)

                if is_learning:
                    exp = trajectory_lib.from_transition(
                        time_step, action_step._replace(action=action), next_time_step
                    )
                    replay_buffer.add_batch(exp)

                    if replay_buffer.num_frames().numpy() >= warmup_frames:
                        try:
                            experience, _ = next(dataset_iter)
                        except StopIteration:
                            dataset_iter = iter(dataset)
                            experience, _ = next(dataset_iter)
                        loss_info = agent.train(experience)
                        li = float(loss_info.loss.numpy())
                        episode_losses.append(li)
                        if step % 1000 == 0:
                            print(f"[Ep {episode}] step {step} | loss {li:.4f}")

                    if hasattr(agent.collect_policy, "decay_epsilon"):
                        agent.collect_policy.decay_epsilon(steps=1)

                time_step = next_time_step

            if last_slate is None:
                last_slate = tf.zeros((num_users, slate_size), dtype=tf.int32)
                last_choice = tf.fill((num_users,), slate_size)
                last_reward = tf.zeros((num_users,), dtype=tf.float32)

            relevance = tf.zeros_like(last_slate, dtype=tf.float32)
            click_mask = tf.logical_and(last_choice >= 0, last_choice < slate_size)
            idx = tf.where(click_mask)
            relevance = tf.tensor_scatter_nd_update(
                relevance,
                tf.concat([tf.cast(idx, tf.int32), tf.expand_dims(tf.gather(last_choice, idx[:, 0]), 1)], axis=1),
                tf.ones((tf.shape(idx)[0],), dtype=tf.float32)
            )

            last_slate_np = last_slate.numpy()
            relevance_np = relevance.numpy()
            click_mask_np = click_mask.numpy()
            ndcg = ndcg_at_k(last_slate_np, relevance_np, k=slate_size)
            mrr = slate_mrr(last_slate_np, relevance_np, k=slate_size)
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            click_rate = float(np.mean(click_mask_np))
            episode_reward = float(episode_reward_t.numpy())
            eps_out = None
            if is_learning and hasattr(agent.collect_policy, "epsilon"):
                try:
                    eps_out = float(agent.collect_policy.epsilon)
                except Exception:
                    eps_out = None

            payload = {
                "episode": int(episode),
                "total_reward": float(episode_reward),
                "loss": float(avg_loss) if is_learning else 0.0,
                "ndcg@5": float(ndcg),
                "slate_mrr": float(mrr),
                "click_rate": float(click_rate),
            }
            if eps_out is not None:
                payload["epsilon"] = eps_out

            logger.log(payload)
            if eps_out is not None:
                print(f"[Episode {episode}] total reward = {episode_reward:.2f} | click_rate = {click_rate:.3f} | ε={eps_out:.3f}")
            else:
                print(f"[Episode {episode}] total reward = {episode_reward:.2f} | click_rate = {click_rate:.3f}")

            del episode_losses, last_slate, last_reward, last_choice
            if episode % 20 == 0:
                gc.collect()
                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    pass

        if FLAGS.peek_final_state:
            trajectory_result = rt.trajectory()
            user_state = trajectory_result.get('user_state')
            if user_state:
                interest = user_state.get('interest')
                if isinstance(interest, tf.Tensor):
                    for i, value in enumerate(interest.numpy()):
                        print(f"User {i} final interest: {value}")
                else:
                    print("Interest:", interest)
            else:
                print("No user_state in final trajectory")

    finally:
        logger.close()

        if FLAGS.plot:
            try:
                metrics_df = pd.read_csv(logger.csv_file)

                if "episode" not in metrics_df.columns:
                    return
                metrics_df["episode"] = pd.to_numeric(metrics_df["episode"], errors='coerce')
                metrics_df = metrics_df.dropna(subset=["episode"])
                metrics_df["episode"] = metrics_df["episode"].astype(int)

                for col in ["total_reward", "loss", "ndcg@5", "slate_mrr", "click_rate", "epsilon"]:
                    if col in metrics_df.columns:
                        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')

                log_dir_path = logger.log_dir if isinstance(logger.log_dir, Path) else Path(logger.log_dir)
                run_name = log_dir_path.name
                plots_dir = Path("plots") / agent_name
                plots_dir.mkdir(parents=True, exist_ok=True)

                plt.style.use("default")

                ep_series = metrics_df["episode"]
                xticks = ep_series[ep_series % 100 == 0].astype(int).tolist()

                def plot_with_avg(df, ycol, title, ylabel, filename, ma_window=20,
                                main_color="#0C00AD", ma_color="#9ABDFF"):
                    if ycol not in df.columns or not df[ycol].notnull().any():
                        return
                    x = df["episode"].values
                    y = df[ycol].values
                    y_valid = y[~np.isnan(y)]
                    if y_valid.size == 0:
                        return
                    y_mean = float(np.mean(y_valid))
                    y_ma = pd.Series(y).rolling(ma_window, min_periods=1).mean().values

                    plt.figure()
                    plt.plot(x, y, label=ycol, color=main_color)
                    plt.plot(x, y_ma, label=f"MA({ma_window})", linestyle='-.', linewidth=1.25, color=ma_color)
                    plt.axhline(y=y_mean, linestyle='--', alpha=0.7, label=f"Mean = {y_mean:.3f}", color=ma_color)
                    plt.xlabel("Episode"); plt.ylabel(ylabel)
                    plt.title(f"{title} — {agent_name}")
                    if xticks:
                        plt.xticks(xticks)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.legend()
                    plt.savefig(plots_dir / f"{run_name}_{filename}.png")
                    plt.close()

                plot_with_avg(metrics_df, "total_reward",
                            "Total Reward over Episodes", "Total Reward", "reward",
                            main_color="#0C00AD", ma_color="#9ABDFF")
                plot_with_avg(metrics_df, "loss",
                            "Training Loss over Episodes", "Loss", "loss",
                            main_color="#B59AFF", ma_color="#9ABDFF")

                has_ndcg = "ndcg@5" in metrics_df and metrics_df["ndcg@5"].notnull().any()
                has_mrr  = "slate_mrr" in metrics_df and metrics_df["slate_mrr"].notnull().any()
                if has_ndcg or has_mrr:
                    plt.figure()
                    x = metrics_df["episode"].values

                    if has_ndcg:
                        y = metrics_df["ndcg@5"].values
                        y_ma = pd.Series(y).rolling(20, min_periods=1).mean().values
                        y_mean = float(np.nanmean(y))
                        plt.plot(x, y, label="NDCG@5", color="#0C00AD")
                        plt.plot(x, y_ma, label="NDCG@5 MA(20)", linestyle='-.', linewidth=1.25, color="#9ABDFF")
                        plt.axhline(y=y_mean, linestyle='--', alpha=0.7, label=f"NDCG mean={y_mean:.3f}", color="#9ABDFF")

                    if has_mrr:
                        y2 = metrics_df["slate_mrr"].values
                        y2_ma = pd.Series(y2).rolling(20, min_periods=1).mean().values
                        y2_mean = float(np.nanmean(y2))
                        plt.plot(x, y2, label="Slate MRR", color="#B59AFF")
                        plt.plot(x, y2_ma, label="MRR MA(20)", linestyle='-.', linewidth=1.25, color="#FF9ACD")
                        plt.axhline(y2_mean, linestyle='--', alpha=0.7, label=f"MRR mean={y2_mean:.3f}", color="#FF9ACD")

                    plt.xlabel("Episode"); plt.ylabel("Ranking score")
                    plt.title(f"Ranking Metrics over Episodes — {agent_name}")
                    if xticks:
                        plt.xticks(xticks)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.legend()
                    plt.savefig(plots_dir / f"{run_name}_ranking.png")
                    plt.close()
            
                has_click = "click_rate" in metrics_df and metrics_df["click_rate"].notnull().any()
                has_eps   = "epsilon" in metrics_df and metrics_df["epsilon"].notnull().any()
                if has_click or has_eps:
                    plt.figure()
                    x = metrics_df["episode"].values

                    if has_click:
                        y = metrics_df["click_rate"].values
                        y_ma = pd.Series(y).rolling(20, min_periods=1).mean().values
                        y_mean = float(np.nanmean(y))
                        
                        plt.plot(x, y, label="Click Rate", color="#0C00AD")
                        plt.plot(x, y_ma, label="Click Rate MA(20)", linestyle='-.', linewidth=1.25, color="#9ABDFF")
                        plt.axhline(y=y_mean, linestyle='--', alpha=0.7, label=f"Click mean={y_mean:.3f}", color="#9ABDFF")

                    if has_eps:
                        y2 = metrics_df["epsilon"].values
                        y2_ma = pd.Series(y2).rolling(20, min_periods=1).mean().values
                        y2_mean = float(np.nanmean(y2))

                        plt.plot(x, y2, label="Epsilon", color="#B59AFF")
                        plt.plot(x, y2_ma, label="Eps MA(20)", linestyle='-.', linewidth=1.25, color="#FF9ACD")
                        plt.axhline(y2_mean, linestyle='--', alpha=0.7, label=f"Eps mean={y2_mean:.3f}", color="#FF9ACD")

                    plt.xlabel("Episode"); plt.ylabel("Value")
                    plt.title(f"Click Rate (and Epsilon) over Episodes — {agent_name}")
                    if xticks:
                        plt.xticks(xticks)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.legend()
                    plt.savefig(plots_dir / f"{run_name}_click_epsilon.png")
                    plt.close()

            except Exception:
                print("(Plotting skipped due to an error)")

if __name__ == '__main__':
    app.run(main)
