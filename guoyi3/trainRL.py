# -*- coding: utf-8 -*-
from worker.self_play_win import SelfPlayWorker
import os
import config_rl as cfg
from collections import defaultdict
from agent.model import CChessModel
from agent.player import CChessPlayer, State
cur_path = os.path.dirname(__file__)
os.chdir(cur_path)
print("working dir: ", cur_path)


def main():
    network = CChessModel(cfg)
    if cfg.pretrained_model is not None:
        network.restore(cfg.pretrained_model)
    player = CChessPlayer(cfg, defaultdict(State), pv_fn=network.eval)




if __name__ == "__main__":
    main()
