'''
Description: 用于调试和与随机棋手PK使用的源代码
Author: ouyhlan
Date: 2021-01-05 15:29:15
'''
from Board import Board
from Game import Othello
from Mcts import MCTS
from Net import OthelloNet
import argparse
import torch
import numpy as np

class RandomPlayer():
    def __init__(self, game, color):
        self.game = game
        self.color = color
    
    def play(self):
        legal_moves = self.game.GetPlayerLegalMoves(self.color)
        action = legal_moves[np.random.randint(len(legal_moves))]
        return action

class HumanPlayer():
    def __init__(self, game, color):
        self.game = game
        self.color = color
    
    def play(self):
        legal_moves = self.game.GetPlayerLegalMoves(self.color)

        while True:
            print(f"legal_moves:{legal_moves}")
            input_move = input('Please enter your move(in the form of (x, y)):')
            
            if input_move == '' or eval(input_move) not in legal_moves:
                print("Invalid move!")
            else:
                return eval(input_move)
                

    
class BetaDogPlayer():
    def __init__(self, args, game, color):
        self.args = args
        self.game = game
        self.color = color
        self.net = OthelloNet(args).to(args.cuda)
        self.mcts = MCTS(args, self.game, self.net)
        
    def play(self):
        n = self.args.n
        board = self.game.chessboard.chessboard

        # sample action from improved policy
        pi = self.mcts.GetNextActionProb(board, self.color)
        action = np.random.choice(len(pi), p=pi)
        move = (int(action / n), action % n)
        return move
    
    def Reset(self, game):
        self.game = game
        self.mcts = MCTS(self.args, self.game, self.net)
        
    def FromPretrained(self, it):
        dir = args.model_dir + f'/{it}.pth'
        state = torch.load(dir)
        self.net.load_state_dict(state['net'])
        self.mcts = MCTS(self.args, self.game, self.net)

def ParseOpt():
    '''
        Hyper-Parameters
    '''
    parser = argparse.ArgumentParser(description = 'For Beta Dog')
    parser.add_argument('--examples_dir', type = str, default = 'Examples_log', help = 'Folder directory for saving examples')
    parser.add_argument('--model_dir', type = str, default = 'Model_log', help = 'Folder directory for saving models')
    parser.add_argument('--n', type = int, default = 8, help = 'Chessboard size n * n')
    parser.add_argument('--num_iters', type = int, default = 1000, help = 'Number of self-play simulations')
    parser.add_argument('--num_episodes', type = int, default = 100, help = 'Number of complete self-play games to simualte in an iteration')
    parser.add_argument('--max_example_size', type = int, default = 200000, help = 'Number of examples to train the neural networks')
    parser.add_argument('--num_example_history', type = int, default = 20, help = 'Keep example for several times')
    parser.add_argument('--num_of_mcts_sim', type = int, default = 200, help = 'Number of MCTS simulations times')
    parser.add_argument('--temp_threshold', type = int, default = 0, help = 'temperature threshold')
    parser.add_argument('--c_puct', type = float, default = 1.0, help = 'Const in PUCT algorithm')
    parser.add_argument('--filter_num', type = int, default = 256, help = 'Number of filter in Neural Network')
    parser.add_argument('--filter_size', type = int, default = 3, help = 'Kernel size of Cnn filter')
    parser.add_argument('--res_layer_num', type = int, default = 10, help = 'Number of residual blocks')
    parser.add_argument('--cuda', type = str, default = 'cuda:0', help = 'Pytorch Cuda')
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Training Batch Size')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Training Epochs')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Learning Rate')
    parser.add_argument('--dropout', type = float, default = 0.3, help = 'Dropout rate')
    parser.add_argument('--seed', type = int, default = 123, help = 'Random seed for initialization')

    return parser.parse_args()

args = ParseOpt()
game = Othello()
human_black_player = HumanPlayer(game, -1)
human_white_player = HumanPlayer(game, 1)
random_black_player = RandomPlayer(game, -1)
random_white_player = RandomPlayer(game, 1)
beta_dog_black_player = BetaDogPlayer(args, game, -1)
beta_dog_black_player.FromPretrained(18) # 设置beta-dog版本号 版本号越高 理论上越强 现在版本最新为4
beta_dog_white_player = BetaDogPlayer(args, game, 1)
beta_dog_white_player.FromPretrained(18) # 设置beta-dog版本号 版本号越高 理论上越强

game.PrintBoard()
ag_wins, random_wins = 0, 0
for i in range(50):
    game = Othello()
    random_black_player = RandomPlayer(game, -1)
    beta_dog_white_player.Reset(game)
    
    while game.turn != 0:
        # 设置黑色棋手
        if game.turn == -1:
            action = random_black_player.play()
            print(f'Black plays in {action}')
        # 设置白色棋手
        else:
            action = beta_dog_white_player.play()
            print(f'White plays in {action}')

        game.Play(action)
        game.PrintBoard()
        wc, bc = game.CurrentScore()
        print(f"#White : #Black = {wc, bc}")
        
    if wc > bc:
        ag_wins += 1
    elif bc > wc:
        random_wins += 1
    #logger.info(f"Current score: ag {ag_wins} : {random_wins} random")
#logger.info(f"Final score: ag {ag_wins} : {random_wins} random")