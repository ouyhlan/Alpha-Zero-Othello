from Mcts import MCTS
from Game import Othello
from collections import deque
from Net import OthelloNet
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np

class OthelloDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.datas = dataset

    def __getitem__(self, index):
        example = self.datas[index]
        return {'state':example[0], 'pi':example[1], 'z':example[2]}

    def __len__(self):
        return len(self.datas)

def my_collate(batch):
    state, pi, z = [], [], []
    for item in batch:
        state.append(item['state'].tolist())
        pi.append(item['pi'])
        z.append(item['z'])
    return torch.FloatTensor(state), torch.FloatTensor(pi), torch.FloatTensor(z)

class SelfPlay():
    '''
    Self-play + Learning
    '''

    def __init__(self, args, net):
        self.args = args
        self.net = net
        self.train_history = []
        self.curr_it = 1
        self.num_cores = self.args.num_of_cores

    @staticmethod
    def ExecuteEpisode(args, net):
        '''
            Self-play a match, starting with player Black
        '''
        n = args.n
        training_history = []
        game = Othello()
        mcts = MCTS(args, game, net)
        episode_step = 0

        while game.turn != 0:
            episode_step += 1

            # Get Training Example
            board = game.chessboard.chessboard
            s = Othello.GetStandardBoard(board, game.turn)
            temp = int(episode_step < args.temp_threshold)
            
            # Symmetries
            pi = mcts.GetNextActionProb(board, game.turn, temp)
            sym_list = Othello.GetSymmetries(s, pi)

            for b, p in sym_list:
                training_history.append([b, game.turn, p])

            # Self-play
            action = np.random.choice(len(pi), p=pi)
            move = (int(action / n), action % n)
            game.Play(move)

        final_board = game.chessboard.chessboard
        r = Othello.GetFinalReward(final_board, 1)
        return [(x[0], x[2], r * ((-1) ** (x[1] != 1))) for x in training_history]
    
    @staticmethod
    def Multi(proc_num, args, curr_iter, iter_times, return_list):
        dir = args.model_dir + f'/{curr_iter - 2}.pth'
        state = torch.load(dir)
        net = OthelloNet(args).to(args.cuda)
        net.load_state_dict(state['net'])
        
        print(f'{proc_num} starts!')
        res = []
        for i in tqdm(range(iter_times), desc = f'Process #{proc_num} Self Play'):
            res += SelfPlay.ExecuteEpisode(args, net)
        return_list.append(res)
        print(f'{proc_num} ends!')

    def Learn(self):
        '''
        Perform several self-play
        '''
        for i in range(self.curr_it, self.args.num_iters + 1):
            print(f'Starting Iter #{i} ...')
            examples_queue = deque([], maxlen = self.args.max_example_size)
            
            '''
            for _ in tqdm(range(self.args.num_episodes), desc = 'Self Play'):
                    examples_queue += SelfPlay.ExecuteEpisode(self.args, self.net)
            '''
            
            # Parallel Self-play training
            manager = mp.Manager()
            return_list = manager.list()
            self.processes = []

            extra = self.args.num_episodes % self.num_cores
            for qk in range(1, self.num_cores + 1):
                iter_times = int(self.args.num_episodes / self.num_cores) +  1 * (qk <= extra)
                self.processes.append(mp.Process(target=SelfPlay.Multi, args = (qk, self.args, i, iter_times, return_list)))
            
            # process stuff
            [process.start() for process in self.processes]
            [process.join() for process in self.processes]
            [process.terminate() for process in self.processes]
            [process.join() for process in self.processes]

            for qk in range(self.num_cores):
                examples_queue += return_list[qk]

            # save the iteration examples to the history
            self.train_history.append(list(examples_queue))

            if len(self.train_history) > self.args.num_example_history:
                print(f"Removing the oldest example in iteration {i}")
                self.train_history.pop(0)
            
            self.SaveExamples(i - 1)
            train_data = []
            for history in self.train_history:
                train_data.extend(history)

            train_dataset = OthelloDataset(self.args, train_data)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.args.batch_size, collate_fn=my_collate, shuffle = True)
            self.Train(train_loader)
            self.SaveNet(i - 1)

    def SaveExamples(self, it):
        if not os.path.exists(self.args.examples_dir):
            os.makedirs(self.args.examples_dir)
        dir = self.args.examples_dir + f'/{it}.examples'
        torch.save(self.train_history, dir)

    def LoadExamples(self, it):
        dir = self.args.examples_dir + f'/{it}.examples'
        self.curr_it = it + 2
        self.train_history = torch.load(dir)

    def SaveNet(self, it):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        dir = self.args.model_dir + f'/{it}.pth'
        state = {'net':self.net.state_dict()}
        torch.save(state, dir)

    def LoadNet(self, it):
        dir = self.args.model_dir + f'/{it}.pth'
        state = torch.load(dir)
        self.net.load_state_dict(state['net'])

    def Train(self, train_loader):
        optimizer = optim.Adam(self.net.parameters())
        
        for epoch in range(self.args.epochs):
            print(f'epoch {epoch + 1}:')
            self.net.train()

            train_loss = 0
            for data in tqdm(train_loader, desc = 'Traing Neural Net'):
                optimizer.zero_grad()
                state, pi, z = data

                p, v = self.net(state.to(self.args.cuda))
                p, v = F.log_softmax(p, dim=1).cpu(), v.view(-1).cpu()

                mse_loss = nn.MSELoss()
                ce_loss = -torch.sum(p * pi) / p.size()[0]
                loss = mse_loss(z, v) + ce_loss
                train_loss += loss

                loss.backward()
                optimizer.step()
            
            print(f"batch_size: {self.args.batch_size} | num_of_batch: {len(train_loader)} | loss: {train_loss / len(train_loader)}")