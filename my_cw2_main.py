from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work
from cw2.cw_data.cw_wandb_logger import WandBLogger

import main
from utils.get_agent import get_new_ppo_agent

def create_wandb_dict(metrics_dict: dict, rewards_dict: dict):
    wandb_dict = metrics_dict.copy()
    wandb_dict.update(rewards_dict)
    return wandb_dict

class MyExperiment(experiment.AbstractIterativeExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        params = config["params"]
        # necessary if config.yml contains grid or list instead of params, because exp-names have to be different
        exp_name = config["path"].split("/")[-1]
        params['exp_name'] = exp_name
        config['iterations'] = params['train_steps']
        self.agent = get_new_ppo_agent(params)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        metrics_dict, rewards_dict = self.agent.step()
        if self.agent.save_interval > 0 and n % self.agent.save_interval == 0:
            self.agent.save(n)
        wandb_dict = create_wandb_dict(metrics_dict, rewards_dict)
        return wandb_dict

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass
        # Skip for Quickguide
        if not crash:
            self.agent.save(self.agent.train_steps)

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass


if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    # Optional: Add loggers
    cw.add_logger(WandBLogger())

    # RUN!
    cw.run()