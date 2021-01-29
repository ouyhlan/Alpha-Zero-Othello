'''
Description: 神经网络源代码
Author: ouyhlan
Date: 2021-01-05 15:29:15
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, args):
        super(BasicBlock, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(
                        in_channels = args.filter_num, 
                        out_channels = args.filter_num, 
                        kernel_size = args.filter_size, 
                        stride = 1,
                        padding = 1
                    )
        self.conv2 = nn.Conv2d(
                        in_channels = args.filter_num, 
                        out_channels = args.filter_num, 
                        kernel_size = args.filter_size, 
                        stride = 1,
                        padding = 1
                    )
        
        self.bn1 = nn.BatchNorm2d(args.filter_num)
        self.bn2 = nn.BatchNorm2d(args.filter_num)
    
    def forward(self, x):
        # x (batch_size, 1, n, n)
        #x = x.view(-1, self.args.filter_num, self.args.n, self.args.n)
        
        in_x = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return F.relu(in_x + x)

class OthelloNet(nn.Module):
    def __init__(self, args):
        super(OthelloNet, self).__init__()

        self.args = args
        self.action_size = args.n * args.n + 1
        self.conv1 = nn.Conv2d(
                                in_channels = 1, 
                                out_channels = args.filter_num, 
                                kernel_size = args.filter_size, 
                                stride = 1,
                                padding = 1
                            )
        self.conv2 = nn.Conv2d(
                                in_channels = args.filter_num,
                                out_channels = 2,
                                kernel_size = 1,
                                stride = 1
                            )
        self.conv3 = nn.Conv2d(
                                in_channels =  args.filter_num,
                                out_channels = 1,
                                kernel_size = 1,
                                stride = 1
                            )

        self.bn1 = nn.BatchNorm2d(args.filter_num)
        self.bn2 = nn.BatchNorm2d(2)
        self.bn3 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(2 * args.n * args.n, args.n * args.n + 1)
        self.fc2 = nn.Linear(args.n * args.n, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.residual_blocks = nn.ModuleList([BasicBlock(args) for _ in range(args.res_layer_num)])
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # x (batch_size, 1, n, n)
        x = x.view(-1, 1, self.args.n, self.args.n)
        batch_size = x.shape[0]
        
        # Input layer
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Tower
        for i in range(self.args.res_layer_num):
            x = self.residual_blocks[i](x)

        # Output layer
        # Policy
        # policy_out (batch_size, 2 * n * n)
        policy_out = F.relu(self.bn2(self.conv2(x))).view(batch_size, -1)
        
        # policy_out (batch_size, n * n + 1)
        policy_out = self.fc1(policy_out)

        # Value
        # value_out (batch_size, n, n)
        value_out = F.relu(self.bn3(self.conv3(x))).view(batch_size, -1)

        # value_out (batch_size, 256)
        value_out = self.dropout(F.relu(self.fc2(value_out)))
        
        # value_out (batch_size, 1)
        value_out = self.fc3(value_out)

        return policy_out, torch.tanh(value_out)
