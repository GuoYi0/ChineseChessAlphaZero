# -*- coding: utf-8 -*-
from agent.player import CChessPlayer, State
import config_rl as cfg
from collections import defaultdict
from environment import static_env as senv

class MCTS(object):
    def __init__(self, config, net, simulation_per_step):
        self.player = CChessPlayer(cfg, defaultdict(State))
        pass

    def run(self):
        state = senv.INIT_STATE
        history = [state]
        game_over = False
        turns = 0
        no_act = None
        increase_temp = False

        while not game_over:
            action, policy = self.player.action(state, turns, no_act, increase_temp=increase_temp)


        pass


