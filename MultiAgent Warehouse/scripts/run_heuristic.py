import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
from tarware.heuristic import heuristic_episode
import tarware

import warnings
warnings.filterwarnings('ignore')

import os
import csv
from datetime import datetime

parser = ArgumentParser(description="Run A* Heuristic", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--size", default="tiny", choices=["tiny", "small", "medium", "large", "extralarge"], help="Warehouse size")
parser.add_argument("--num_agvs", default=3, type=int, choices=range(1, 20), help="Number of AGVs")
parser.add_argument("--num_pickers", default=2, type=int, choices=range(1, 10), help="Number of pickers")
parser.add_argument("--obs_type", default="partial", choices=["partial", "global"], help="Observation type")
parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes")

args = parser.parse_args()

def info_statistics(infos, global_episode_return, episode_returns):
    _total_deliveries = 0
    _total_clashes = 0
    _total_stuck = 0
    for info in infos:
        _total_deliveries += info["shelf_deliveries"]
        _total_clashes += info["clashes"]
        _total_stuck += info["stucks"]
        info["total_deliveries"] = _total_deliveries
        info["total_clashes"] = _total_clashes
        info["total_stuck"] = _total_stuck
    last_info = infos[-1]
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = global_episode_return
    last_info["episode_returns"] = episode_returns
    return last_info

if __name__ == "__main__":
    tarware.full_registration()
    env_id = f"tarware-{args.size}-{args.num_agvs}agvs-{args.num_pickers}pickers-{args.obs_type}obs-v1"
    print(f"Running environment: {env_id}")
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", f"{env_id}_summary.csv")
    env = gym.make(env_id)
    seed = args.seed
    completed_episodes = 0
    results = []

    for i in range(args.num_episodes):
        start = time.time()
        infos, global_episode_return, episode_returns = heuristic_episode(env.unwrapped, seed = seed+i)
        end = time.time()
        last_info = info_statistics(infos, global_episode_return, episode_returns)
        last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
        last_info["episode"] = completed_episodes
        last_info["env"] = f"{args.size}-{args.num_agvs}agvs-{args.num_pickers}pickers-{args.obs_type}obs"
        last_info["id"] = f"{args.seed+i}_ep{completed_episodes}"
        last_info["time_taken"] = end - start
        results.append(last_info)
        episode_length = len(infos)
        
        print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}]")
        completed_episodes += 1

    fieldnames = [
    "pickers_distance_travelled",
    "pickers_idle_time",
    "vehicles_busy",
    "shelf_deliveries",
    "agvs_idle_time",
    "clashes",
    "agvs_distance_travelled",
    "stucks",
    "total_deliveries",
    "total_clashes",
    "total_stuck",
    "episode_length",
    "global_episode_return",
    "episode_returns",
    "overall_pick_rate",
    "episode",
    "env",
    "id",
    "time_taken",
]


    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {csv_path}")
