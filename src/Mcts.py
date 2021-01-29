'''
Description: 蒙特卡洛树搜索算法实现源代码
Author: ouyhlan
Date: 2021-01-05 18:29:15
'''
import math
import torch
import numpy as np
import sys  # 导入sys模块
from Game import Othello
import torch.nn.functional as F

sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

class MCTS():
    '''
    parameters:
        s -- str state of board
        a -- int action, count by #column * n + #row  n * n means no valid move
    '''
    def __init__(self, args, game, net):
        '''
        N(s, a) W(s, a) Q(s, a) P(s, a)
        '''
        self.args = args
        self.game = game
        self.net = net
        
        self.W_s = {}   # Since neural net return vector, use vector representation
        self.N_s = {}   # Since neural net return vector, use vector representation
        self.P_s = {}   # Since neural net return vector, use vector representation

        # Store Visited Situation
        self.end_situation = {}
        self.valid_set = {}

    def GetNextActionProb(self, board, player, tau=1):
        '''
        Assuming that temp
        input:
            board -- current board state
            player -- current player
        output:
            prob -- probability of each action
        '''
        n = self.args.n
        standard_board = Othello.GetStandardBoard(board, player)
        s = Othello.Board2String(standard_board)
        
        for i in range(self.args.num_of_mcts_sim):
            self.Search(standard_board)
        
        counts = self.N_s[s].numpy()

        if tau == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            pi = np.zeros(n * n + 1)
            pi[best_action] = 1
            return pi
        
        #counts = [self.N_s[s][a] ** (1. / tau) if N_s.get(s) is not None else 0 for a in range(n * n)]
        counts = counts ** (1. / tau)
        return counts / counts.sum()


    def Search(self, standard_board):
        '''
        action size = n * n + 1 (1 for do nothing)
        input:
            standard_board -- alway for the white to move
        output:
        '''
        n = self.args.n
        s = Othello.Board2String(standard_board)

        if s not in self.end_situation:
            self.end_situation[s] = Othello.GetFinalReward(standard_board, 1)
        if self.end_situation[s] is not None:
            return -self.end_situation[s]
        
        # Expand and Evaluate
        if self.P_s.get(s) is None:
            net_input = torch.FloatTensor(standard_board.astype(np.float64)).to(self.args.cuda)
            with torch.no_grad():
                self.net.eval()
                self.P_s[s], v = self.net(net_input)
            self.P_s[s], v = F.softmax(self.P_s[s], dim=1).view(-1).cpu(), v.cpu().item()

            valid_mask = Othello.GetValidMoves(standard_board, 1)
            self.P_s[s] = self.P_s[s] * valid_mask

            self.N_s[s] = torch.zeros([n * n + 1])
            self.W_s[s] = torch.zeros([n * n + 1])
            self.valid_set[s] = valid_mask
            return -v
        
        s_valid = self.valid_set[s]
        max_qu = -math.inf
        max_a = -1

        # Select
        for a in range(n * n + 1):
            # Compute valid action
            if s_valid[a]:
                # PUCT algorithm
                Q_sa = self.W_s[s][a] / self.N_s[s][a] if self.N_s[s][a] !=0 else 0
                U_sa = self.args.c_puct * self.P_s[s][a] * math.sqrt(self.N_s[s].sum()) / (1 + self.N_s[s][a])

                res = Q_sa + U_sa
                if res > max_qu:
                    max_qu = res
                    max_a = a
        
        next_s, next_player = Othello.GetNextState(standard_board, 1, max_a)
        next_s = Othello.GetStandardBoard(next_s, next_player)  # Get Standard board for next search

        # Backup
        v = self.Search(next_s)
        self.W_s[s][max_a] += v
        self.N_s[s][max_a] += 1

        return -v