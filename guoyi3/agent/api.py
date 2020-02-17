# -*- coding: utf-8 -*-
class CChessModelAPI(object):
    def __init__(self, config, agent_model):
        self.agent_model = agent_model
        self.pipes = []  # 用于交互
        self.config = config
