# SlateQ-RecSys-Research

This repository contains my dissertation project at the University of Bath for the MSc Computer Science degree. It implements and evaluates slate-based reinforcement learning algorithms in a recommender system simulation environment, built on [**RecSim NG**](https://github.com/google-research/recsim_ng) (Google’s next-generation simulator for recommender system research) [[Mladenov et al., 2021]](https://arxiv.org/abs/2103.08057).

The project investigates the [**SlateQ algorithm**](https://www.ijcai.org/proceedings/2019/0360.pdf) [[Ie et al., 2019]](https://www.ijcai.org/proceedings/2019/0360.pdf) and its variants, comparing them against simpler baselines such as random, greedy, and contextual bandit approaches. The primary focus is on analysing long-term user dynamics, evaluating ranking quality, and assessing the ability of learning agents to outperform short-sighted heuristics.


---

## Project Structure

```
SlateQ-RecSys-Research/                 # Project root
├── main.py                             # Entry point: training loop, logging, plotting
├── experiments/                        # Experiment configuration files
│   └── configs/
│       └── base.gin                    # Gin config for environment and agent parameters
├── src/                                # Source code
│   ├── agents/                         # Implemented agent baselines and RL algorithms
│   │   ├── random_agent.py             # Random slate selection
│   │   ├── greedy_agent.py             # Greedy baseline (affinity only)
│   │   ├── ctxbandit_agent.py          # Contextual bandit baseline
│   │   ├── slateq_agent.py             # Vanilla SlateQ agent
│   │   ├── slateq_dueling_agent.py     # SlateQ with dueling Q-network
│   │   ├── slateq_noisynet_agent.py    # SlateQ with noisy layers for exploration
│   │   └── dqn_agent.py                # Vanilla DQN agent (non-slate baseline)
│   ├── entities/                       # Core RecSim NG entity/state models
│   │   ├── ecomm_user.py               # User model with long-term interest dynamics
│   │   ├── ecomm_item.py               # Item catalogue with static features
│   │   └── ecomm_recommender.py        # Recommender agent state model
│   ├── stories/                        # Environment wiring
│   │   └── ecomm_story.py              # Connects user, item, recommender, and response
│   ├── runtimes/                       # Custom runtime wrappers
│   │   └── ecomm_runtime.py            # Runtime loop built on RecSim NG’s TFRuntime
│   ├── metrics/                        # Evaluation and logging utilities
│   │   ├── logger.py                   # Logs metrics to CSV and JSONL
│   │   └── ranking_metrics.py          # NDCG@K and slate MRR implementations
│   └── core/                           # Framework glue code
│       └── registry.py                 # Agent registry for easy instantiation
├── logs/                               # Auto-generated logs of training runs
│   └── <agent_name>/                   # Separate folder for each agent
│       └── run_YYYY_MM_DD_HH_MM/       # Run-specific directory
│           ├── metrics.csv             # Metrics logged in CSV
│           └── metrics.jsonl           # Metrics logged in JSON Lines
└── plots/                              # Auto-generated training plots
    └── <agent_name>/                   # Separate folder for each agent
        └── run_YYYY_MM_DD_HH_MM/       # Run-specific plots
            ├── reward.png              # Reward vs episodes
            ├── loss.png                # Loss vs episodes
            ├── ranking.png             # NDCG@K and MRR vs episodes
            └── click_epsilon.png       # Click rate and epsilon vs episodes
```

---

## Installation

This project assumes Python 3.10+.

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd SlateQ-RecSys-Research
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running Experiments

Experiments are launched through `main.py`, which accepts both configuration files and agent names.

Command to run:
```bash
python main.py --gin_files=experiments/configs/base.gin [agent name]
```

Example run with SlateQ:
```bash
python main.py --gin_files=experiments/configs/base.gin slateq
```

Available agent names:
- `random`
- `greedy`
- `ctxbandit`
- `slateq`
- `slateqdueling`
- `slateqnoisynet`
- `dqn`

---

## Logging

Each run creates a directory under `logs/` with the timestamp of the run. Inside, you will find:

- `metrics.csv`: tabular metrics per episode (reward, loss, NDCG, MRR, click rate, epsilon).
- `metrics.jsonl`: the same metrics in JSON Lines format for flexible parsing.

---

## Plotting

At the end of training, plots are automatically generated in `plots/<agent_name>/run_<timestamp>/`.

Generated figures include:
- `reward.png`: Total reward vs episodes
- `loss.png`: Training loss vs episodes
- `ranking.png`: NDCG@5 and Slate MRR vs episodes
- `click_epsilon.png`: Click rate and exploration epsilon vs episodes

The plots include mean lines and moving averages for smoother visualisation.

---

## Notes

- Random seeds are fixed (NumPy and TensorFlow) for reproducibility.
- TensorFlow threading is limited to reduce nondeterminism.
- The user state model is designed to ensure non-myopic, long-horizon behaviour.

---

## References

- Mladenov, M., Ie, E., et al. (2021). **RecSim NG: Toward Principled Uncertainty Modeling for Recommender Ecosystems**. arXiv preprint [arXiv:2103.08057](https://arxiv.org/abs/2103.08057).
- Ie, E., et al. (2019). **SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets**. Proceedings of IJCAI 2019. [PDF](https://www.ijcai.org/proceedings/2019/0360.pdf).

---

_Muhammad Rasyid Gatra Wijaya - MSc Computer Science - University of Bath - 249389034_
