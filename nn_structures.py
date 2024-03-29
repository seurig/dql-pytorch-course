''' Collection of simple Neural Network architectures for use in Reinforcement Learning '''

import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    ''' Simple Neural Network for use in Reinforcement Learning '''
    def __init__(self, input_dims, output_dims, lr, weights_file):
        ''' Neural Network creation using:
        - input_dims
        - output_dims
        - lr
        - weights_file
        '''

        super(NeuralNet, self).__init__()
        self.weights_file = weights_file

        self.fc1 = nn.Linear(*input_dims, 512)
        self.fc2 = nn.Linear(512, output_dims)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        if os.path.isfile(self.weights_file):
            self.load_weights()

    def forward(self, data):
        ''' predict from given data '''
        layer1 = F.relu(self.fc1(data))
        layer2 = self.fc2(layer1)

        return layer2

    def load_weights(self):
        ''' loading weights '''
        self.load_state_dict(T.load(self.weights_file))

    def save_weights(self):
        ''' saving weights '''
        T.save(self.state_dict(), self.weights_file)

class DuelingNeuralNet(nn.Module):
    ''' Simple Dueling Neural Network for use in Reinforcement Learning '''
    def __init__(self, input_dims, output_dims, lr, weights_file):
        ''' Neural Network creation using:
        - input_dims
        - output_dims
        - lr
        - weights_file
        '''
        super(DuelingNeuralNet, self).__init__()
        self.weights_file = weights_file

        self.fc1 = nn.Linear(*input_dims, 512)
        self.fc2 = nn.Linear(512, 512)

        self.V_layer = nn.Linear(512, 1)
        self.A_layer = nn.Linear(512, output_dims)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        if os.path.isfile(self.weights_file):
            self.load_weights()

    def forward(self, data):
        ''' predict from given data '''
        layer1 = F.relu(self.fc1(data))
        layer2 = F.relu(self.fc2(layer1))

        V = self.V_layer(layer2)
        A = self.A_layer(layer2)

        return V, A

    def load_weights(self):
        ''' loading weights '''
        self.load_state_dict(T.load(self.weights_file))

    def save_weights(self):
        ''' saving weights '''
        T.save(self.state_dict(), self.weights_file)

class ConvNeuralNet(nn.Module):
    ''' Convolutional Neural Network for use in Atari Reinforcement Learning '''
    def __init__(self, input_dims, output_dims, lr, weights_file):
        ''' Neural Network creation using:
        - input_dims
        - output_dims
        - lr
        - weights_file
        '''
        super(ConvNeuralNet, self).__init__()
        self.weights_file = weights_file

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.calculate_fc_input_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, output_dims)

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        if os.path.isfile(self.weights_file):
            self.load_weights()

    def calculate_fc_input_dims(self, input_dims):
        ''' calculating size use first fully connected layer after feature extraction'''
        tnsr = T.zeros(1, *input_dims)
        dims = self.conv1(tnsr)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, data):
        ''' predict from given data '''
        conv1 = F.relu(self.conv1(data))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = conv3.view(conv3.size()[0], -1)
        fc1 = F.relu(self.fc1(flatten))
        fc2 = self.fc2(fc1)

        return fc2

    def load_weights(self):
        ''' loading weights '''
        self.load_state_dict(T.load(self.weights_file))

    def save_weights(self):
        ''' saving weights '''
        T.save(self.state_dict(), self.weights_file)

class DuelingConvNeuralNet(nn.Module):
    ''' Dueling Convolutional Neural Network for use in Atari Reinforcement Learning '''
    def __init__(self, input_dims, output_dims, lr, weights_file):
        ''' Neural Network creation using:
        - input_dims
        - output_dims
        - lr
        - weights_file
        '''
        super(DuelingConvNeuralNet, self).__init__()
        self.weights_file = weights_file

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.calculate_fc_input_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)

        self.V_layer = nn.Linear(512, 1)
        self.A_layer = nn.Linear(512, output_dims)

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        if os.path.isfile(self.weights_file):
            self.load_weights()

    def calculate_fc_input_dims(self, input_dims):
        ''' calculating size use first fully connected layer after feature extraction'''
        tnsr = T.zeros(1, *input_dims)
        dims = self.conv1(tnsr)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, data):
        ''' predict from given data '''
        conv1 = F.relu(self.conv1(data))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = conv3.view(conv3.size()[0], -1)

        fc1 = F.relu(self.fc1(flatten))
        V = self.V_layer(fc1)
        A = self.A_layer(fc1)

        return V, A

    def load_weights(self):
        ''' loading weights '''
        self.load_state_dict(T.load(self.weights_file))

    def save_weights(self):
        ''' saving weights '''
        T.save(self.state_dict(), self.weights_file)
