# -*- coding: utf-8 -*-
from collections import deque
from agent.model import CChessModel
import pandas as pd
from logging import getLogger
from time import time
from environment.env import CChessEnv
from environment.lookup_tables import ActionLabelsRed, flip_policy, flip_move
import numpy as np

logger = getLogger(__name__)
"""有监督学习模型"""


class SupervisedWorker(object):
    """
    监督学习模型
    """

    def __init__(self, cfg):
        self.config = cfg
        self.model = CChessModel(self.config)  # 神经网络class
        self.dataset = deque(), deque(), deque()  # 局面，行动，价值
        self.buffer = []  # 数据缓存
        self.gameinfo = None
        self.moves = None
        self.env = CChessEnv(self.config)

    def start(self):
        # 载入数据，监督学习的大师的标签
        self.gameinfo = pd.read_csv(self.config.Train.sl_data_gameinfo)
        self.moves = pd.read_csv(self.config.Train.sl_data_move)
        self.training()

    def load_game(self, red, black, winner, idx):
        """
        :param red: 红方所有move
        :param black: 黑方所有move
        :param winner: "red" or "black"
        :param idx:
        :return:
        """
        self.env.reset()
        red_moves = []
        black_moves = []
        turns = 1
        black_max_turn = black['turn'].max()  # 走棋步数
        red_max_turn = red['turn'].max()

        while turns < black_max_turn or turns < red_max_turn:
            if turns < red_max_turn:
                wxf_move = red[red.turn == turns]['move'].item()  # 形如r6+4的一步棋
                action = self.env.board.parse_WXF_move(wxf_move)  # 解析走法 形如 r6+4 => 3478表示 (3,4)位置去棋子走到(7,8)
                try:
                    # env.observation是一个字符串表示的局面，self.build_policy返回一个np.zeros(2048)，只有一个位置是1
                    red_moves.append([self.env.observation, self.build_policy(action, flip=False)])
                except Exception as e:
                    for i in range(10):
                        logger.debug(f"{self.env.board.screen[i]}")
                    logger.debug(f"{turns} {wxf_move} {action}")
                self.env.step(action)  # 走一步
            if turns < black_max_turn:
                wxf_move = black[black.turn == turns]['move'].item()
                action = self.env.board.parse_WXF_move(wxf_move)
                try:
                    black_moves.append([self.env.observation, self.build_policy(action, flip=True)])
                except Exception as e:
                    for i in range(10):
                        logger.debug(f"{self.env.board.screen[i]}")
                    logger.debug(f"{turns} {wxf_move} {action}")
                self.env.step(action)
            turns += 1
        if winner == 'red':
            red_win = 1
        elif winner == 'black':
            red_win = -1
        else:
            red_win = 0
        for move in red_moves:
            move += [red_win]
        for move in black_moves:
            move += [-red_win]
        data = []
        for i in range(len(red_moves)):
            data.append(red_moves[i])
            if i < len(black_moves):
                data.append(black_moves[i])
        self.buffer += data

    def convert_to_trainging_data(self):
        data = self.buffer
        state_list = []
        policy_list = []
        value_list = []
        for state_fen, policy, value in data:
            # state_fen是一个字符串表示的局面，policy是一个np.zero(2048)只有一个是1，value是+1或者0或者-1
            state_planes = self.env.fen_to_planes(state_fen)  # 返回np.zeros(shape=(14, 10, 9))的输入特征
            sl_value = value

            state_list.append(state_planes)
            policy_list.append(policy)
            value_list.append(sl_value)

        return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(
            value_list, dtype=np.float32)

    def generate_game_data(self, games):
        self.buffer = []
        start_time = time()
        for idx, game in games.iterrows():  # 读取一局游戏
            gid = game['gameID']  # 57380690
            winner = game['winner']  # red or black
            move = self.moves[self.moves.gameID == gid]  # 取出该游戏下所有的动作，也是一个pandas对象
            red = move[move.side == 'red']  # 红方的所有move
            black = move[move.side == 'black']  # 黑方的所有move
            self.load_game(red, black, winner, idx)
        end_time = time()
        logger.debug(f"Loading {len(games)} games, time: {end_time - start_time}s")
        return self.convert_to_trainging_data()

    def fill_queue(self, games):
        print("begin to fill data")
        _tuple = self.generate_game_data(games)  # 返回 （局面，行动，价值）三元组
        if _tuple is not None:
            for x, y in zip(self.dataset, _tuple):
                x.extend(y)
        print("fill data done!")

    def build_policy(self, action, flip):
        labels_n = len(ActionLabelsRed)
        move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
        policy = np.zeros(labels_n)
        policy[move_lookup[action]] = 1
        if flip:
            policy = flip_policy(policy)
        return policy

    def training(self):
        logger.info(
            f"Start training, game count = {len(self.gameinfo)}, step = {self.config.Train.sl_game_step} games")
        # 下面两个值都是 1w
        gameinfo_len = len(self.gameinfo)
        total_loop = gameinfo_len // self.config.Train.sl_game_step * self.config.Train.num_epoch
        loop = 0
        for _ in range(self.config.Train.num_epoch):
            for i in range(0, gameinfo_len, self.config.Train.sl_game_step):
                begin_time = time()
                games = self.gameinfo[i:min(i + self.config.Train.sl_game_step, gameinfo_len)]
                self.fill_queue(games)  # 填充数据
                self.train_epoch(self.config.Train.batch_size)
                a, b, c = self.dataset
                a.clear()
                b.clear()
                c.clear()
                end_time = time()
                loop += 1
                print("%d/%d time cost: %d" % (loop, total_loop, int(end_time - begin_time)))

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = self.dataset
        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def train_epoch(self, batch_size):
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        # state_ary.shape 是（棋局数*棋长度，14， 10， 9）；policy_ary.shape（棋局数*棋长度，2086）
        # valu_ary.shape是（棋局数*棋长度，）
        # 分别表示当前局面，基于当前局面的走法，当前局面落子之前的价值
        idx = 0
        data_length = state_ary.shape[0]
        while idx + batch_size <= data_length:
            step, xentropy_loss, value_loss, total_loss = self.model.train_one_step(
                state_ary[idx: idx + batch_size], policy_ary[idx: idx + batch_size], value_ary[idx: idx + batch_size])
            print("step %d, policy_loss %0.3f, value_loss %0.3f, total_loss %0.3f" % (
                step, xentropy_loss, value_loss, total_loss))
            idx += batch_size
        # 最后一条数据别浪费了
        if idx < data_length:
            step, xentropy_loss, value_loss, total_loss = self.model.train_one_step(
                state_ary[idx: data_length], policy_ary[idx: data_length], value_ary[idx: data_length])
            print("step %d, policy_loss %0.3f, value_loss %0.3f, total_loss %0.3f" % (
                step, xentropy_loss, value_loss, total_loss))
