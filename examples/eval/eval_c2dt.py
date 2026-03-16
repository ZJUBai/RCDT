from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import sys
import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field
from osrl.algorithms import CDT, CDTQCTWeightedTrainer, Critic
from osrl.common.exp_util import load_config_and_model, seed_all

# ========= 核心：每个任务的 returns / costs 硬编码 =========
TASK_CONFIG = {
    # Bullet Safety Gym
    "OfflineCarCircle-v0":  {"returns": [450, 500, 550], "costs": [10, 20, 40]},
    "OfflineAntRun-v0":     {"returns": [700, 750, 800], "costs": [10, 20, 40]},
    "OfflineDroneRun-v0":   {"returns": [400, 500, 600], "costs": [10, 20, 40]},
    "OfflineDroneCircle-v0":{"returns": [700, 750, 800], "costs": [10, 20, 40]},
    "OfflineCarRun-v0":     {"returns": [575, 575, 575], "costs": [10, 20, 40]},
    "OfflineAntCircle-v0":  {"returns": [300, 350, 400], "costs": [10, 20, 40]},
    "OfflineBallCircle-v0": {"returns": [500, 500, 500], "costs": [10, 20, 40]},
    "OfflineBallRun-v0":    {"returns": [700, 750, 800], "costs": [10, 20, 40]},

    # Safety Gymnasium: Car
    "OfflineCarButton1Gymnasium-v0": {"returns": [35, 35, 35], "costs": [20, 40, 80]},
    "OfflineCarButton2Gymnasium-v0": {"returns": [40, 40, 40], "costs": [20, 40, 80]},
    "OfflineCarCircle1Gymnasium-v0": {"returns": [20, 22.5, 25], "costs": [20, 40, 80]},
    "OfflineCarCircle2Gymnasium-v0": {"returns": [20, 21, 22], "costs": [20, 40, 80]},
    "OfflineCarGoal1Gymnasium-v0": {"returns": [40, 40, 40], "costs": [20, 40, 80]},
    "OfflineCarGoal2Gymnasium-v0": {"returns": [30, 30, 30], "costs": [20, 40, 80]},
    "OfflineCarPush1Gymnasium-v0": {"returns": [15, 15, 15], "costs": [20, 40, 80]},
    "OfflineCarPush2Gymnasium-v0": {"returns": [12, 12, 12], "costs": [20, 40, 80]},

    # Safety Gymnasium: Point
    "OfflinePointButton1Gymnasium-v0": {"returns": [40, 40, 40], "costs": [20, 40, 80]},
    "OfflinePointButton2Gymnasium-v0": {"returns": [40, 40, 40], "costs": [20, 40, 80]},
    "OfflinePointCircle1Gymnasium-v0": {"returns": [50, 52.5, 55], "costs": [20, 40, 80]},
    "OfflinePointCircle2Gymnasium-v0": {"returns": [45, 47.5, 50], "costs": [20, 40, 80]},
    "OfflinePointGoal1Gymnasium-v0": {"returns": [30, 30, 30], "costs": [20, 40, 80]},
    "OfflinePointGoal2Gymnasium-v0": {"returns": [30, 30, 30], "costs": [20, 40, 80]},
    "OfflinePointPush1Gymnasium-v0": {"returns": [15, 15, 15], "costs": [20, 40, 80]},
    "OfflinePointPush2Gymnasium-v0": {"returns": [12, 12, 12,], "costs": [20, 40, 80]},

    # Velocity tasks (higher returns)
    "OfflineAntVelocityGymnasium-v1": {"returns": [2800, 2800, 2800], "costs": [20, 40, 80]},
    "OfflineHalfCheetahVelocityGymnasium-v1": {"returns": [3000, 3000, 3000], "costs": [20, 40, 80]},
    "OfflineHopperVelocityGymnasium-v1": {"returns": [1750, 1750, 1750], "costs": [20, 40, 80]},
    "OfflineSwimmerVelocityGymnasium-v1": {"returns": [160, 160, 160], "costs": [20, 40, 80]},
    "OfflineWalker2dVelocityGymnasium-v1": {"returns": [2800, 2800, 2800], "costs": [20, 40, 80]},
}



@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    # returns: List[float] = field(default=[1000,1000,1000], is_mutable=True)
    # costs: List[float] = field(default=[10, 20, 40], is_mutable=True)
    returns: List[float] = field(default=[400,450,500,550,600,650,700,750,800,850,900,950,1000], is_mutable=True)
    costs: List[float] = field(default=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], is_mutable=True)
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = True
    device: str = "cuda"
    threads: int = 4
    seed: int = 0


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)
    # seed_all(cfg["seed"])
    seed_all(args.seed)

    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])

    target_entropy = -env.action_space.shape[0]

    # model & optimizer & scheduler setup
    cdt_model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=cfg["embedding_dim"],
        seq_len=cfg["seq_len"],
        episode_len=cfg["episode_len"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        attention_dropout=cfg["attention_dropout"],
        residual_dropout=cfg["residual_dropout"],
        embedding_dropout=cfg["embedding_dropout"],
        time_emb=cfg["time_emb"],
        use_rew=cfg["use_rew"],
        use_cost=cfg["use_cost"],
        cost_transform=cfg["cost_transform"],
        add_cost_feat=cfg["add_cost_feat"],
        mul_cost_feat=cfg["mul_cost_feat"],
        cat_cost_feat=cfg["cat_cost_feat"],
        action_head_layers=cfg["action_head_layers"],
        cost_prefix=cfg["cost_prefix"],
        stochastic=cfg["stochastic"],
        init_temperature=cfg["init_temperature"],
        target_entropy=target_entropy,
    )
    q_critic = Critic(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_dim=cfg["embedding_dim"])
    c_critic = Critic(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_dim=cfg["embedding_dim"])

    cdt_model.load_state_dict(model["model_state"])
    cdt_model.to(args.device)
    q_critic.load_state_dict(model["q_critic_state"])
    q_critic.to(args.device)
    c_critic.load_state_dict(model["c_critic_state"])
    c_critic.to(args.device)

    trainer = CDTQCTWeightedTrainer(cdt_model,
                           q_critic,
                           c_critic,
                         env,
                         reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"],
                         cost_reverse=cfg["cost_reverse"],
                         device=args.device)

    # rets = args.returns
    # costs = args.costs
    task_name = cfg["task"]
    rets = TASK_CONFIG[task_name]["returns"]
    costs = TASK_CONFIG[task_name]["costs"]

    assert len(rets) == len(
        costs
    ), f"The length of returns {len(rets)} should be equal to costs {len(costs)}!"
    for target_ret, target_cost in zip(rets, costs):
        seed_all(cfg["seed"])
        ret, cost, length = trainer.evaluate(args.eval_episodes,
                                             target_ret * cfg["reward_scale"],
                                             target_cost * cfg["cost_scale"])
        env.set_target_cost(target_cost)
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        # print(
        #     f"[{task_name}] Target reward {target_ret}, real reward {ret}, normalized reward: {normalized_ret}; target cost {target_cost}, real cost {cost}, normalized cost: {normalized_cost}"
        # ,file=sys.stderr)
        print(
            f"{normalized_ret} {normalized_cost}"
        ,file=sys.stderr)

if __name__ == "__main__":
    eval()
