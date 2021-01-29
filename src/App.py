import argparse
import pygame, sys, random
import torch
import numpy as np
from Game import Othello
from Mcts import MCTS
from Net import OthelloNet
from pygame.locals import *
 
BACKGROUNDCOLOR = (255, 255, 255)
BLACK = (255, 255, 255)
BLUE = (0, 0, 255)
CELLWIDTH = 68
CELLHEIGHT = 68
PIECEWIDTH = 66
PIECEHEIGHT = 66
BOARDX = 27
BOARDY = 16
FPS = 40
 
# 退出
def terminate():
    pygame.quit()
    sys.exit()

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
        temp = 0

        # sample action from improved policy
        pi = self.mcts.GetNextActionProb(board, self.color, temp)
        action = np.random.choice(len(pi), p=pi)
        move = (int(action / n), action % n)
        return move
    
    def Reset(self, game):
        self.game = game
        self.mcts = MCTS(self.args, self.game, self.net)
        
    def FromPretrained(self, it):
        dir = args.model_dir + f'/{it}.pth'
        state = torch.load(dir, map_location=self.args.cuda)
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
    parser.add_argument('--temp_threshold', type = int, default = 15, help = 'temperature threshold')
    parser.add_argument('--c_puct', type = float, default = 1.0, help = 'Const in PUCT algorithm')
    parser.add_argument('--filter_num', type = int, default = 256, help = 'Number of filter in Neural Network')
    parser.add_argument('--filter_size', type = int, default = 3, help = 'Kernel size of Cnn filter')
    parser.add_argument('--res_layer_num', type = int, default = 10, help = 'Number of residual blocks')
    parser.add_argument('--cuda', type = str, default = 'cpu', help = 'Pytorch Cuda')
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Training Batch Size')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Training Epochs')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Learning Rate')
    parser.add_argument('--dropout', type = float, default = 0.3, help = 'Dropout rate')
    parser.add_argument('--seed', type = int, default = 123, help = 'Random seed for initialization')
    parser.add_argument('--beta_dog_version', type = int, default = 26, help = 'Beta dog player version')

    return parser.parse_args()

game = Othello()
args = ParseOpt()

print('There are two mode:\n1. Player Black \n2. Ai Black')
choice = input("Please input your choice:")
human = -1 if eval(choice) == 1 else 1
beta_dog_player = BetaDogPlayer(args, game, -human)
beta_dog_player.FromPretrained(args.beta_dog_version) # 设置beta-dog版本号 版本号越高 理论上越强 现在版本最新为4
print(f'Beta Dog Version : {args.beta_dog_version}')

# 初始化
pygame.init()
mainClock = pygame.time.Clock()
 
# 加载图片
boardImage = pygame.image.load('board.png')
boardRect = boardImage.get_rect()
blackImage = pygame.image.load('black.bmp')
blackRect = blackImage.get_rect()
whiteImage = pygame.image.load('white.bmp')
whiteRect = whiteImage.get_rect()
 
basicFont = pygame.font.SysFont(None, 48)
gameover_str = 'Game Over Score '
 
# 设置窗口
windowSurface = pygame.display.set_mode((boardRect.width, boardRect.height))
pygame.display.set_caption('黑白棋')
 
move_once = False
game_over = False

# 游戏主循环
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            terminate()
        
        # Human's turn
        if move_once is False and game_over is False and game.turn == human and event.type == MOUSEBUTTONDOWN and event.button == 1:
            x, y = pygame.mouse.get_pos()
            col = int((x-BOARDX)/CELLWIDTH)
            row = int((y-BOARDY)/CELLHEIGHT)
            print(col, row)
            game.Play((col, row))
            game.PrintBoard()

            move_once = True
 
    windowSurface.fill(BACKGROUNDCOLOR)
    windowSurface.blit(boardImage, boardRect, boardRect)
 
    if move_once is False and game_over == False and game.turn == -human:
        x, y = beta_dog_player.play()
        game.Play((x, y))
        move_once = True
 
    windowSurface.fill(BACKGROUNDCOLOR)
    windowSurface.blit(boardImage, boardRect, boardRect)
    
    for x in range(args.n):
        for y in range(args.n):
            rectDst = pygame.Rect(BOARDX+x*CELLWIDTH+2, BOARDY+y*CELLHEIGHT+2, PIECEWIDTH, PIECEHEIGHT)
            if game.chessboard[x, y] == -1:
                windowSurface.blit(blackImage, rectDst, blackRect)
            elif game.chessboard[x, y] == 1:
                windowSurface.blit(whiteImage, rectDst, whiteRect)
    move_once = False
    
    if game.turn == 0:
        game_over = True
        white_score, black_score = game.CurrentScore()
        output_str = gameover_str + f' #white {white_score} : {black_score} #black'
        text = basicFont.render(output_str, True, BLACK, BLUE)
        textRect = text.get_rect()
        textRect.centerx = windowSurface.get_rect().centerx
        textRect.centery = windowSurface.get_rect().centery
        windowSurface.blit(text, textRect)
    
    pygame.display.update()
    mainClock.tick(FPS)