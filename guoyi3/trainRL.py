# -*- coding: utf-8 -*-
from worker.self_play_win import SelfPlayWorker
import os
import config_rl as cfg
cur_path = os.path.dirname(__file__)
os.chdir(cur_path)
print("working dir: ", cur_path)


def main():
    rl = SelfPlayWorker(cfg)
    rl.start()


if __name__ == "__main__":
    main()
