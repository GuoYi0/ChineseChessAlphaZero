# -*- coding: utf-8 -*-
import config_rl as cfg
from environment.lookup_tables import ActionLabelsRed, flip_move
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import config_rl as config
from environment import static_env as senv
import numpy as np


class Action(object):  # 动作，对应一条边
    def __init__(self):
        self.n = 0  # N(s, a) : visit count
        self.w = 0  # W(s, a) : total action value
        self.q = 0  # Q(s, a) = N / W : action value
        self.p = 0  # P(s, a) : prior probability


class State(object):  # 状态，对应一个结点，一个棋局
    def __init__(self):
        self.a = defaultdict(Action)  # key: action, value: ActionState
        self.sum_n = 0  # visit count
        self.p = None  # policy of this state，从action index 到概率的映射
        self.legal_moves = None  # all leagal moves of this state
        self.w = 0  # TODO 这个值没有使用


class CChessPlayer(object):
    def __init__(self, cfg=None, training=False, search_tree=None, pv_fn=None):
        self.config = config
        self.pv_fn = pv_fn
        self.training = training
        self.labels_n = len(ActionLabelsRed)
        self.labels = ActionLabelsRed
        # 从move到index的映射
        self.move_lookup = {move: i for move, i in zip(self.labels, range(self.labels_n))}
        self.increase_temp = False
        self.enable_resign = config.enable_resign
        if search_tree is None:
            self.tree = defaultdict(State)  # 键是一个字符串表示的棋局，值是一个State结点
        else:
            self.tree = search_tree
        self.root_state = None
        self.no_act = None  # 有可能是为搜索树进行剪枝的，即不会往这些动作走（及时合法）

    def reset(self, search_tree=None):
        self.tree = defaultdict(State) if search_tree is None else search_tree
        self.root_state = None
        self.no_act = None
        self.increase_temp = False

    def action(self, state, turns, no_act=None):
        """
        从state出发进行树搜索，搜完以后，以确定基于当前局面该怎么走棋
        :param state: 一个字符串表示的棋局
        :param turns:
        :param no_act: 不需要考虑的动作，以免长将或者长捉
        :return:
        """
        self.root_state = state
        # 执行搜索，完毕以后形成了一棵树
        for i in range(self.config.simulation_per_step):
            self.MCTS_search(state, [state])
        policy, resign = self.calc_policy(state, turns, no_act)
        if resign:  # 直接投降
            return None
        if no_act is not None:
            for act in no_act:
                policy[self.move_lookup[act]] = 0
        if not self.training:
            my_action = np.argmax(policy)
        else:
            p = self.apply_temperature(policy, turns)
            my_action = int(np.random.choice(range(self.labels_n), p=p))
        return self.labels[my_action]

    def apply_temperature(self, policy, turns):
        if turns < 30 and self.config.tau_decay_rate != 0:
            tau = np.power(self.config.tau_decay_rate, turns)
        else:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, state, turns, no_act):
        """
        根据visit count计算policy
        :param state:
        :param turns:
        :param no_act:
        :return:
        """
        node = self.tree[state]
        policy = np.zeros(self.labels_n)
        max_q_value = -100

        for mov, action_state in node.a.items():
            policy[self.move_lookup[mov]] = action_state.n
            if no_act and mov in no_act:
                policy[self.move_lookup[mov]] = 0
                continue
            if action_state.q > max_q_value:
                max_q_value = action_state.q
        # 直接投降
        if max_q_value < self.config.resign_threshold and self.enable_resign and turns > self.config.min_resign_turn:
            return policy, True
        policy /= np.sum(policy)
        return policy, False

    def update_tree(self, p, v, history: list):
        """
        :param p: policy 当前局面对红方的策略
        :param v: value, 当前局面对红方的价值
        :param history: 包含当前局面的一个棋局，(state, action) pair
        :return:
        """
        state = history.pop()  # 最近的棋局
        if p is not None:  # 对于展开后的回溯，这句话成立
            node = self.tree[state]
            node.p = p
        #  注意，这里并没有把v赋给当前node
        while len(history) > 0:
            action = history.pop()
            state = history.pop()
            v = -v
            node = self.tree[state]  # 状态结点
            action_state = node.a[action]  # 该状态下的action边
            action_state.n += 1
            action_state.w += v
            action_state.q = action_state.w * 1.0 / action_state.n

    def MCTS_search(self, state, history):
        """
        从当前state出发进行一次搜索至叶子节点
        :param state: 字符串棋局，当前局面， 从当前局面开始搜
        :param history: 装着字符串棋局的列表, 包含当前棋局
        :return:
        """
        while True:
            # v是当前状态对红方的价值
            game_over, v, _ = senv.done(state)
            if game_over:
                v = v * 2
                self.update_tree(None, v, history=history)
                break
            if state not in self.tree:
                self.tree[state].sum_n = 1
                self.tree[state].legal_moves = senv.get_legal_moves(state)  # 展开
                state_planes = senv.state_to_planes(state)
                policy, value = self.pv_fn(state_planes[np.newaxis, ...])  # 返回当前局面对红方的策略和价值
                self.update_tree(policy, value, history)
                break
            if state in history[:-1]:
                # 这个局面以前出现过
                for i in range(len(history) - 1):
                    if history[i] == state:
                        if senv.will_check_or_catch(state, history[i + 1]):
                            self.update_tree(None, -1, history)
                        elif senv.be_catched(state, history[i + 1]):
                            self.update_tree(None, 1, history)
                        else:
                            self.update_tree(None, 0, history)
                        break
                break
            sel_action = self.select_action_q_and_u(state)
            self.tree[state].sum_n += 1
            action_state = self.tree[state].a[sel_action]
            action_state.q = action_state.w / action_state.n
            history.append(sel_action)  # 装上基于当前动作选择的action
            state = senv.step(state, sel_action)
            history.append(state)  # 装上下一个state

    def select_action_q_and_u(self, state) -> str:
        """
        选择最佳动作
        :param state:
        :param is_root_node:
        :return:
        """
        is_root_node = self.root_state == state
        node = self.tree[state]
        legal_moves = node.legal_moves

        # 为合法move分配先验概率
        if node.p is not None:  # policy of this state
            all_p = 0
            for mov in legal_moves:
                mov_p = node.p[self.move_lookup[mov]]
                node.a[mov].p = mov_p
                all_p += mov_p
            # rearrange the distribution
            if all_p == 0:
                all_p = 1
            for mov in legal_moves:
                node.a[mov].p /= all_p
            node.p = None

        xx_ = np.sqrt(node.sum_n + 1)
        e = self.config.noise_eps
        c_puct = self.config.c_puct
        dir_alpha = self.config.dirichlet_alpha

        best_score = -99999999
        best_action = None
        move_counts = len(legal_moves)

        for mov in legal_moves:
            if is_root_node and self.no_act and mov in self.no_act:
                continue
            action_state = node.a[mov]
            p_ = action_state.p  # 该动作的先验概率
            # 只为根节点加噪声
            if is_root_node:
                p_ = (1 - e) * p_ + e * np.random.dirichlet(dir_alpha * np.ones(move_counts))[0]
            score = action_state.q + c_puct * p_ * xx_ / (1 + action_state.n)
            if action_state.q > (1 - 1e-7):  # q值接近于1的，直接作为最佳结点
                best_action = mov
                break
            if score >= best_score:
                best_score = score
                best_action = mov
        return best_action
