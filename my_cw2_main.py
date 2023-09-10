from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work

import main


class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass
        # Skip for Quickguide

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        main.entrypoint_for_cw2(config.get("params"))

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass
        # Skip for Quickguide


if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    # Optional: Add loggers
    #cw.add_logger(...)

    # RUN!
    cw.run()