import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

# linear classifier

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) # self.parameters() comes from nn.Module
        self.loss = nn.CrossEntropyLoss() # nn.MSEloss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # 1st GPU if GPU
        self.to(self.device) # float/int of tensor vs. cuda-tensors for GPU

    def forward(self, data)
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2) # loss fct handles activation

        return layer3

    def learn(self, data, labels):
        self.optimizer.zero_grad() # new gradients for batch
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)
        # .tensor() preserves datatype vs .Tensor()

        predictions = self.forward(data)

        cost = self.loss(predictions, labels) # calculate loss

        cost.backward() # back propagation
        self.optimizer.step() # optimize

