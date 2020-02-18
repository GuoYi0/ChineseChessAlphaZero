# -*- coding: utf-8 -*-
from agent.player import CChessPlayer, State
import config_rl as cfg
from collections import defaultdict
from environment import static_env as senv
from time import time


class MCTS(object):
    def __init__(self, config, net, simulation_per_step):
        self.player = CChessPlayer(cfg, defaultdict(State))
        self.config = cfg
        pass

    def run(self):
        state = senv.INIT_STATE
        history = [state]
        game_over = False
        turns = 0
        no_act = None
        no_eat_count = 0  # 记录没有吃子的局数
        final_move = None
        value = 0

        while not game_over:
            start_time = time()
            action = self.player.action(state, turns, no_act)
            end_time = time()
            if action is None:
                print(f"{turns % 2} (0 = 红; 1 = 黑) 投降了!")
                value = -1
                break
            print(
                f"博弈中: 回合{turns // 2 + 1} {'红方走棋' if turns % 2 == 0 else '黑方走棋'}, 着法: {action}, 用时: {(end_time - start_time):.1f}s")
            history.append(action)
            state, no_eat = senv.new_step(state, action)  # 走一步，并翻转局面，当前局面则是黑方了
            if no_eat:
                no_eat_count += 1
            else:
                no_eat_count = 0
            history.append(state)
            if no_eat_count >= 120 or turns / 2 >= self.config.max_game_length:
                game_over = True
                value = 0
            else:
                # 当前局面的情形。check表示己方是否处于被将军状态
                game_over, value, final_move, check = senv.done(state, need_check=True)
                no_act = []
                if not game_over:
                    if not senv.has_attack_chessman(state):
                        print(f"双方无进攻子力，作和。state = {state}")
                        game_over = True
                        value = 0
                # 游戏没结束，action的行为没有将对方军，并且该局面在以前重复过
                if not game_over and not check and state in history[:-1]:
                    free_move = defaultdict(int)
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            # 判断 state 执行history[i + 1]是否在捉子或者将军
                            if senv.will_check_or_catch(state, history[i + 1]):
                                no_act.append(history[i + 1])  # 不能长将或者长捉
                            elif not senv.be_catched(state, history[i + 1]):
                                free_move[state] += 1
                                if free_move[state] >= 3:
                                    # 作和棋处理
                                    game_over = True
                                    value = 0
                                    print("闲着循环三次，作和棋处理")
                                    break
        if final_move:  # 最后一步是吃将
            history.append(final_move)
            state = senv.step(state, final_move)
            turns += 1
            value = -value
            history.append(state)
        self.player.reset()
        if turns % 2 == 1:
            value = -value












