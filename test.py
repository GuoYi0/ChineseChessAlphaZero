# -*- coding: utf-8 -*-
from enum import Enum

# a = 'rnba5/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABR'
# def fliped_state(state):
#     rows = state.split('/')
#
#     def _swapcase(a):
#         if a.isalpha():
#             return a.lower() if a.isupper() else a.upper()
#         return a
#
#     def _swapall(aa):
#         return "".join([_swapcase(a) for a in aa])
#
#     return "/".join([_swapall(reversed(row)) for row in reversed(rows)])
#
# print(fliped_state(a))

BOARD_WIDTH = 9
BOARD_HEIGHT = 10
def state_to_board(state):
    """
    初始状态'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
    从上往下，上方红子下方黑子，红方小写
    :param state:
    :return:
    """
    board = [['.' for col in range(BOARD_WIDTH)] for row in range(BOARD_HEIGHT)]
    x = 0
    y = 0
    for k in range(0, len(state)):
        ch = state[k]
        if ch == ' ':
            break
        if ch == '/':
            x = 0
            y += 1
        elif '1' <= ch <= '9':
            for i in range(int(ch)):
                board[y][x] = '.'
                x = x + 1
        else:
            board[y][x] = ch
            # board[y][x] = swapcase(ch, s2b=True)  # 大小写转换，并切换成 "rnbakabnr"格式
            x = x + 1
    return board
def board_to_state(board):
    fen = ''
    for i in range(BOARD_HEIGHT):
        c = 0
        for j in range(BOARD_WIDTH):
            if board[i][j] == '.':
                c = c + 1
            else:
                if c > 0:
                    fen = fen + str(c)
                # fen = fen + swapcase(board[i][j])
                fen = fen + board[i][j]
                c = 0
        if c > 0:
            fen = fen + str(c)
        if i < BOARD_HEIGHT-1:
            fen = fen + '/'
    return fen


b = 'rnbak4/9/1c7/p1p1p1p1p/9/9/P1P1P1P2/1C5C1/9/RNBAKABNR'
print(board_to_state(state_to_board(b)))
