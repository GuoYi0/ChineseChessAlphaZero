# -*- coding: utf-8 -*-
import os
import time
pretrained_model = None  # 预训练模型地址
total_step = 1000  # 对弈这么多局数
board_size = 256
simulation_per_step = 400  # 源码取值为100？
init_lr = 1e-2
lr_ = [(100, 5e-5), (200, 5e-4), (1000, 5e-3), (2000, 5e-4), (3000, 5e-5), (100000000, 5e-6)]


noise_eps = 0.25
c_puct = 1.5
dirichlet_alpha = 0.2

# 一个状态下面所有动作的最大q值若低于resign_threshold，并且enable_resign=True,并且对弈回合数大于min_resign_turn，则直接投降
enable_resign = False
resign_threshold = -0.9
min_resign_turn = 30

max_game_length = 100  # 最大会和数，超过了就判为和棋
tau_decay_rate = 0.98  # 温度随着回合数逐渐衰减，因为越往后，每一步棋越关键，需要越尖峰的分布
step_ckpt = 500  # 每500步保存ckpt
log_dir = os.path.join("summary", "SL_summary", "log_" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()))
ckpt_path = os.path.join("ckpt", "SL_ckpt")
def get_lr(step):
    for item in lr_:
        if step < item[0]:
            return item[1]
    return lr_[-1][-1]

