# -*- coding: utf-8 -*-
import guoyi3.config_rl as config
from agent.model import CChessModel
from multiprocessing import Manager
from agent.MCTS import MCTS

class SelfPlayWorker(object):
    # def __init__(self, config):
    def __init__(self, cfg=None):
        self.config = config
        self.model = CChessModel(self.config)
        if self.config.pretrained_model is not None:
            self.model.restore(self.config.pretrained_model)
        self.tree = MCTS(self.model, simulation_per_step=config.simulation_per_step)


    def start(self):
        step = 1
        while step < self.config.total_step:
            self.model.adjust_lr(config.get_lr(step))
            print("")






