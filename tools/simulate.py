import hydra
import numpy as np
import torch
import random
from tqdm import tqdm
from typing import Optional, Sequence

from configs.config import CONFIG_PATH
from vectorworld.simulation.simulator import Simulator
from vectorworld.simulation.idm_policy import IDMPolicy
from vectorworld.utils.viz import generate_video


class PolicyEvaluator:
    """Run a policy in the simulator without evaluation aggregation.

    Keep the original class name and the evaluate_policy / compute_metrics interface.
    To avoid affecting the current call code; only run summaries will be returned here and no more evaluation statistics will be performed.
    """

    def __init__(self, cfg, policy, env, episode_ids: Optional[Sequence[int]] = None):
        """
        Parameters
        ----------
        cfg : cfg.sim
        policy : planner / actor
        env : Simulator
        episode_ids : Optional[Sequence[int]]
            If available, give a global unique ID (video/trajectories) for each episeode
        """
        self.cfg = cfg
        self.policy = policy
        self.env = env
        self.episode_ids = list(episode_ids) if episode_ids is not None else None

        # Keep only run summaries and no more evaluation of the aggregation
        self.num_finished_episodes = 0
        self.total_sim_steps = 0

    def reset(self):
        """Reset runtime counters + random seeds."""
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.num_finished_episodes = 0
        self.total_sim_steps = 0

    def _get_global_episode_id(self, local_idx: int) -> int:
        if self.episode_ids is not None and local_idx < len(self.episode_ids):
            return int(self.episode_ids[local_idx])
        return int(local_idx)

    def compute_metrics(self):
        """Backward-compatible summary interface; no evaluation metrics are computed."""
        summary_dict = {
            "num_episodes": int(self.num_finished_episodes),
            "total_sim_steps": int(self.total_sim_steps),
        }
        summary_str = [
            f"num_episodes: {summary_dict['num_episodes']}",
            f"total_sim_steps: {summary_dict['total_sim_steps']}",
        ]
        return summary_dict, summary_str

    def evaluate_policy(self):
        """Run policy over all scenarios without evaluation aggregation."""
        self.reset()

        for i in tqdm(range(self.env.num_test_scenarios)):
            global_id = self._get_global_episode_id(i)
            print(f"Simulating environment local_idx={i}, global_id={global_id}")

            obs = self.env.reset(i)
            if hasattr(self.policy, "reset"):
                self.policy.reset(obs)

            terminated = False
            info = None
            episode_steps = 0

            for t in range(self.env.steps):
                if getattr(self.cfg, "visualize", False):
                    render_frame = True
                    if getattr(self.cfg, "lightweight", False) and (t % 3 != 0):
                        render_frame = False
                    if render_frame:
                        self.env.render_state(
                            name=f"{global_id}",
                            movie_path=self.cfg.movie_path,
                        )

                action = self.policy.act(obs)
                obs, terminated, info = self.env.step(action)
                episode_steps += 1

                if terminated:
                    break

            # Simulator.stepCould not close temporary folder: %s
            assert terminated and info is not None, "Episode did not terminate properly."

            self.num_finished_episodes += 1
            self.total_sim_steps += episode_steps

            if getattr(self.cfg, "save_trajectory", False):
                self.env.save_trajectory(global_id)

            if getattr(self.cfg, "visualize", False):
                generate_video(
                    name=f"{global_id}",
                    output_dir=self.cfg.movie_path,
                    delete_images=True,
                )

            if getattr(self.cfg, "verbose", False):
                print(f"[Simulation] episode={global_id} finished, steps={episode_steps}")

        return self.compute_metrics()


def run_serial(cfg):
    torch.manual_seed(cfg.sim.seed)
    random.seed(cfg.sim.seed)
    np.random.seed(cfg.sim.seed)

    env = Simulator(cfg)

    if cfg.sim.policy == "rl":
        raise NotImplementedError("RLPolicy not yet implemented")
    else:
        policy = IDMPolicy(cfg, env)

    evaluator = PolicyEvaluator(cfg.sim, policy, env)
    summary_dict, summary_str = evaluator.evaluate_policy()

    print("\n[Simulation Done]")
    for line in summary_str:
        print(line)

    return summary_dict, summary_str


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    run_serial(cfg)


if __name__ == "__main__":
    main()