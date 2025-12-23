import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import logging

from meld_graph.experiment import Experiment
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
