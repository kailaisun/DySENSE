import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
