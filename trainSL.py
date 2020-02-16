# -*- coding: utf-8 -*-
"""
监督学习
"""
from worker.sl import SupervisedWorker
import config_sl as cfg


def main():
    sl = SupervisedWorker(cfg)
    sl.start()


if __name__ == "__main__":
    main()
