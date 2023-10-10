from collections import deque

import numpy as np
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work
from cw2.cw_data.cw_wandb_logger import WandBLogger

import main
from utils.get_agent import get_new_ppo_agent

def create_wandb_dict(metrics_dict: dict, rewards_dict: dict):
    wandb_dict = metrics_dict.copy()
    wandb_dict.update(rewards_dict)

    wandb_dict_sorted = {}
    exp_eva = ["exploration", "evaluation"]
    constraint = ["entropy", "kl", "constraint_max", "entropy_max", "entropy_diff", "constraint", "entropy_diff_max",
                  "mean_constraint_max", "kl_max", "mean_constraint", "cov_constraint", "cov_constraint_max"]
    loss = ["vf_loss", "trust_region_loss", "entropy_loss", "policy_loss", "loss"]

    for e in wandb_dict:
        # print(e + " " + str(wandb_dict[e]))
        if e in constraint:
            wandb_dict_sorted["constraint" + "/" + e] = wandb_dict[e]
            continue
        if e in loss:
            wandb_dict_sorted["loss" + "/" + e] = wandb_dict[e]
            continue
    for name in exp_eva:
        for e_exp in wandb_dict[name]:
            wandb_dict_sorted[name + "/" + e_exp] = wandb_dict[name][e_exp]

    return wandb_dict_sorted

class MyExperiment(experiment.AbstractIterativeExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        params = config["params"]
        params['seed'] = rep
        # necessary if config.yml contains grid or list instead of params, because exp-names have to be different
        exp_name = config["path"].split("/")[-1]
        params['exp_name'] = exp_name
        config['iterations'] = params['train_steps']
        self.agent = get_new_ppo_agent(params)
        self._last_5_rews = deque(maxlen=5)
        self._last_5_rews_test = deque(maxlen=5)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        metrics_dict, rewards_dict = self.agent.step()
        if self.agent.save_interval > 0 and n % self.agent.save_interval == 0:
            self.agent.save(n)
        self._last_5_rews.append(rewards_dict['exploration']['mean'])
        self._last_5_rews_test.append(rewards_dict['evaluation']['mean'])
        wandb_dict = create_wandb_dict(metrics_dict, rewards_dict)
        return wandb_dict

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        if not crash:
            self.agent.save(self.agent.train_steps)
            self.agent.store["final_results"].append_row({
                'iteration': self.agent.train_steps,
                '5_rewards': np.array(self._last_5_rews).mean(),
                '5_rewards_test': np.array(self._last_5_rews_test).mean(),
            })  #### the following policy evaluation in Fabian's learn() method do not have an effect
            self.agent.store.close()

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass


if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    # Optional: Add loggers
    cw.add_logger(WandBLogger())

    # RUN!
    cw.run()
    print('clusterworks: done')