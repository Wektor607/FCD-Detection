import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import argparse
import logging

# import meld_graph
# import meld_graph.models
from meld_graph.experiment import Experiment
# import meld_graph.dataset
from meld_graph.paths import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Train model using config in config_file
        """
    )
    parser.add_argument(
        "--config_file",
        help="path to experiment_config.py",
        default="config_files/experiment_config.py",
    )
    parser.add_argument("--wandb_logging", action="store_true", help="enable wandb logging.")
    args = parser.parse_args()

    config = load_config(args.config_file)

    # create experiment
    exp = Experiment(config.network_parameters, config.data_parameters, verbose=logging.INFO)
    # train the model
    exp.train()
