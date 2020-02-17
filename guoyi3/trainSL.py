# -*- coding: utf-8 -*-
"""
监督学习
"""
from worker.sl import SupervisedWorker
import config_sl as cfg
import os

cur_path = os.path.dirname(__file__)
os.chdir(cur_path)
print("working dir: ", cur_path)


def main():
    sl = SupervisedWorker(cfg)
    sl.start()


if __name__ == "__main__":
    main()
