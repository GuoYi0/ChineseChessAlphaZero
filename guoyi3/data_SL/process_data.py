# -*- coding: utf-8 -*-
import os
game = {}
"""
处理数据，提出太短的或者不合法的棋局
"""
with open("E:\ChineseChessAlphaZero\guoyi2\data_SL\moves.csv","r") as f:
    lines_move = f.readlines()
for line in lines_move[1:]:
    l = line.strip().split(',')
    ids, player = l[0], l[2]
    if ids in game:
        if player in game[ids]:
            game[ids][player] += 1
        else:
            game[ids][player] = 1
    else:
        game[ids] = {player: 1}

game_info = set()
with open("E:\ChineseChessAlphaZero\guoyi2\data_SL\gameinfo.csv", "r") as f:
    lines_info = f.readlines()

for line in lines_info[1:]:
    l = line.strip().split(',')[0]
    if l not in game_info:
        game_info.add(l)
    else:
        print("duplicate: ", l)

valid_game = game_info.copy()
for g in game_info:
    if g not in game:
        print("move file does not contains, ", g)
        valid_game.remove(g)
    else:
        if "red" not in game[g] or "black" not in game[g]:
            print("only one side in game, ", g)
            valid_game.remove(g)
        else:
            if game[g]["red"] < 9 or game[g]["black"] < 9:
                print("short game, ", g)
                valid_game.remove(g)
            elif not(game[g]["red"] == game[g]["black"] or game[g]["red"] == game[g]["black"]+1):
                print("invalid game, ", g)
                valid_game.remove(g)
new_moves = open("E:\ChineseChessAlphaZero\guoyi2\data_SL\\new_moves.csv","w")
new_moves.write(lines_move[0])
for line in lines_move[1:]:
    ids = line.strip().split(',')[0]
    if ids in valid_game:
        new_moves.write(line)
new_moves.close()

new_gameinfo = open("E:\ChineseChessAlphaZero\guoyi2\data_SL\\new_gameinfo.csv", "w")
new_gameinfo.write(lines_info[0])
for line in lines_info[1:]:
    ids = line.strip().split(',')[0]
    if ids in valid_game:
        new_gameinfo.write(line)
new_gameinfo.close()
print("total valid game:", len(valid_game))


