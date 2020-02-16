# -*- coding: utf-8 -*-
"""监督学习的相关参数"""
import os
import time


class Train(object):
    """训练相关参数
    每条棋局大概0.25个step（batch_size=256)，1w个棋局对应 2500个step
    """
    start_total_steps = 0  # 初始step记为0
    sl_data_gameinfo = "data_SL/gameinfo.csv"  # 有一万条棋局
    sl_data_move = "data_SL/moves.csv"
    sl_game_step = 100  # 每次导入100条棋局进去训练。原来的1w太大了，内存吃不消
    batch_size = 256
    step_ckpt = 500  # 每500步保存ckpt
    num_epoch = 10  # 训练10个epoch
    lr_step = 5000  # 每两个epoch学习率下降0.3
    init_lr = 0.01  # 初始学习率
    lr_decay = 0.3
    log_dir = os.path.join("summary", "SL_summary", "log_" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()))
    ckpt_path = os.path.join("ckpt", "SL_ckpt")


class Game(object):
    """游戏相关的参数"""
    light = True  # 取true会快一些
