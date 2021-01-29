'''
Description: 训练调用的源代码
Author: ouyhlan
Date: 2021-01-06 00:31:00
'''
from SelfPlay import SelfPlay
from Net import OthelloNet
import argparse
import logging
import torch.multiprocessing as mp

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
    parser.add_argument('--cuda', type = str, default = 'cuda:0', help = 'Pytorch Cuda')
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Training Batch Size')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Training Epochs')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Learning Rate')
    parser.add_argument('--dropout', type = float, default = 0.3, help = 'Dropout rate')
    parser.add_argument('--num_of_cores', type = int, default = 7, help = 'cpu cores using for self-play')
    parser.add_argument('--train_checkpoint', type = int, default = 3, help = 'Restore from previous training')

    return parser.parse_args()

if __name__== "__main__":
    mp.set_start_method('spawn')
    args = ParseOpt()
    net = OthelloNet(args).to(args.cuda)
    self_play = SelfPlay(args, net)

    if args.train_checkpoint >= 0:
        print(f'Loading checkpoint {args.train_checkpoint}...')
        self_play.LoadExamples(args.train_checkpoint)
        self_play.LoadNet(args.train_checkpoint)
    #self_play.Train()
    print('Starting to self-play training!')
    self_play.Learn()
