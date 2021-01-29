'''
Description: 黑白棋棋盘源代码
Author: ouyhlan
Date: 2021-01-04 21:30:25
'''
import numpy as np
'''
(x, y) indicates 
    row y column x

Board details:
    1 = white, 0 = empty, -1 = black
'''
class Board():
    # all 8 possible directions on the board
    __directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]

    def __init__(self, n):
        '''
        self.n indicates n * n chessboard
        '''
        self.n = n
        self.chessboard = np.zeros((self.n, self.n), dtype=np.int8)

        # set up initial state
        # white (3, 3) (4, 4)
        # black (4, 3) (3, 4)
        self.chessboard[int(self.n / 2) - 1, int(self.n / 2) - 1] = 1
        self.chessboard[int(self.n / 2), int(self.n / 2)] = 1
        self.chessboard[int(self.n / 2), int(self.n / 2) - 1] = -1
        self.chessboard[int(self.n / 2) - 1, int(self.n / 2)] = -1
    
    def __getitem__(self, index):
        return self.chessboard[index]

    def __IncrementMove(self, move, direction):
        '''
        从其它代码借鉴而来的一种快速计算方法
        '''
        move = tuple(map(sum, zip(move, direction)))

        while all(map(lambda x: 0 <= x < self.n, move)):
            yield move
            move = tuple(map(sum, zip(move,  direction)))

    def __DiscoverLegalMove(self, origin, direction):
        '''
        input:
            origin -- (x, y) current position
            direction -- direction to search
        output:
            legal moves -- if exists else return None
        '''
        color = self[origin]
        flips = False

        for pos in self.__IncrementMove(origin, direction):
            if self[pos] == 0: 
                if flips:
                    return pos
                else:
                    return None
            if self[pos] == color:
                return None
            if self[pos] == -color:
                flips = True
        return None

    def __GetFlips(self, origin, direction, color):
        '''
        input:
            origin -- (x, y) current position
            direction -- direction to search
            color -- current player
        output:
            legal moves -- if exists else return None
        '''
        res = []

        for pos in self.__IncrementMove(origin, direction):
            if self[pos] == 0:
                return  []
            if self[pos] == -color:
                res.append(pos)
            elif self[pos] == color:
                return res
        
        return []

    def CountPieces(self):
        '''
        return (#white, #black)
        '''
        white_count, black_count = 0, 0

        for y in range(self.n):
            for x in range(self.n):
                if self[x, y] == 1:
                    white_count += 1
                elif self[x, y] == -1:
                    black_count += 1
        
        return (white_count, black_count)

    def GetMovesForPos(self, pos):
        '''
        input:
            pos -- (x, y) a position in chessboard
        output:
            list of legal moves
            [] indicates no legal moves
        '''
        (x, y) = pos
        color = self[pos]

        # Skip empty position
        if color == 0:
            return None

        # Search every directions
        available_moves = []
        for direction in self.__directions:
            move = self.__DiscoverLegalMove(pos, direction)
            if move is not None:
                available_moves.append(move)
        
        return available_moves

    def GetLegalMoves(self, color):
        '''
        input:
            color -- current color to move
        output:
            all legal moves
            None if there is no legel moves
        '''
        moves = set()

        for y in range(self.n):
            for x in range(self.n):
                if self[x, y] == color:
                    new_moves = self.GetMovesForPos((x, y))
                    moves.update(new_moves)
        return list(moves)
    
    def ExecuteMove(self, move, color):
        '''
        input:
            move -- place to move
            color -- which color you are in the chessboard
        output:
            True -- Execute successfully
            False -- Failed
        '''
        if self[move] != 0:
            return False

        self.chessboard[move] = color
        pos2filps = [flip for direction in self.__directions
                          for flip in self.__GetFlips(move, direction, color)]
        
        #assert len(pos2filps) > 0
        for pos in pos2filps:
            self.chessboard[pos] = color
        return True