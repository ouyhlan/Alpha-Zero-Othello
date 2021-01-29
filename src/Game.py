'''
Description: 黑白棋游戏源代码
Author: ouyhlan
Date: 2021-01-05 13:29:15
'''
from Board import Board
import numpy as np
import torch

class Othello():
    square_content = {
        -1: "*",
        +0: " ",
        +1: "O"
    }

    def __init__(self, n=8):
        self.n = n    
        self.chessboard = Board(n)
        self.turn = -1
        self.next_legal_moves = self.chessboard.GetLegalMoves(self.turn)

    def GetPlayerLegalMoves(self, color):
        '''
        input:
            ensure that current move is correct
        output:
            all legal moves
            None if there is no legel moves
        '''
        assert color == self.turn
        return self.next_legal_moves


    def Play(self, move):
        '''
        input:
            move position
        output:
            next turn -- 
                1 for white -1 for black 0 for terminal
        '''
        self.chessboard.ExecuteMove(move, self.turn)
        self.turn = -self.turn

        self.next_legal_moves = self.chessboard.GetLegalMoves(self.turn)
        if len(self.next_legal_moves) == 0:
            self.turn = -self.turn
        else:
            return self.turn

        self.next_legal_moves = self.chessboard.GetLegalMoves(self.turn)
        if len(self.next_legal_moves) == 0:
            self.turn = 0
            return 0
        else:
            return self.turn

    def CurrentScore(self):
        return self.chessboard.CountPieces()

    def GetActionSize(self):
        return self.n * self.n + 1

    def PrintBoard(self):
        print("    ", end="")
        for x in range(self.n):
            print(x, end="   ")
        print("\n  " + "-" * (4 * self.n + 1))
        for y in range(self.n):
            print(y, "|", end="")
            for x in range(self.n):
                if (x, y) in self.next_legal_moves:
                    print(f"{x} {y}", end="|")
                else:
                    pieces = self.chessboard[x, y]
                    print(" " + self.square_content[pieces], end= " |")
            print("\n  " + "-" * (4 * self.n + 1))
            #print("\n  ------------------   ")

    # Following staticmethod
    # board -- np.array((n * n))
    @staticmethod
    def Board2String(board):
        return board.tostring()
    
    @staticmethod
    def GetStandardBoard(board, player):
        return board * player
    
    @staticmethod
    def GetFinalReward(board, player):
        '''
        input:
            board -- current board state
            player -- current player
        output:
            reward -- 
                0 if game is drawn
                1 if white wins
                -1 if black wins
                None if game not ended 
        '''
        n = board.shape[0]
        new_board = Board(n)
        new_board.chessboard = np.copy(board)

        if len(new_board.GetLegalMoves(player)) != 0:
            return None
        if len(new_board.GetLegalMoves(-player)) != 0:
            return None
        
        white_count, black_count = new_board.CountPieces()
        if white_count > black_count:
            return 1 * player
        elif white_count < black_count:
            return -1 * player
        else:
            return 0

    @staticmethod
    def GetNextState(board, player, action):
        '''
        input:
            board -- current board state
            player -- current player
            action -- current move == n*n if there is no valid move
        output:
            new_board -- board state after move
            player -- next player
        '''
        n = board.shape[0]
        move = (int(action / n), action % n)
        
        if action == n * n:
            return (board, -player)
        
        new_board = Board(n)
        new_board.chessboard = np.copy(board)
        new_board.ExecuteMove(move, player)
        
        return (new_board.chessboard, -player)
    
    @staticmethod
    def GetValidMoves(board, player):
        '''
        input:
            board -- current board state
            player -- current player
            action -- current move == n*n if there is no valid move
        output:
            new_board -- board state after move
            player -- next player
        '''
        n = board.shape[0]
        valid_move = torch.zeros((n * n + 1)) # 1 if valid

        # Initialize new board    
        new_board = Board(n)
        new_board.chessboard = np.copy(board)
        legal_moves = new_board.GetLegalMoves(player)

        if len(legal_moves) == 0:
            valid_move[-1] = 1
        for x, y in legal_moves:
            valid_move[x * n + y] = 1
        return valid_move
    
    @staticmethod
    def GetSymmetries(board, pi):
        n = board.shape[0]
        pi_board_shape = np.reshape(pi[:-1], (n, n))
        res = []

        for i in range(1, 5):
            for j in [True, False]:
                new_board = np.rot90(board, i)
                new_pi = np.rot90(pi_board_shape, i)

                if j:
                    new_board = np.fliplr(new_board)
                    new_pi = np.fliplr(new_pi)

                    res += [(new_board, list(new_pi.ravel()) + [pi[-1]])]
        return res