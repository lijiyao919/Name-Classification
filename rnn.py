import torch
import torch.nn as nn

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, num_layers, n_category):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, n_category)
        self.softmax = nn.LogSoftmax(dim=1)
        self.num_layer = num_layers
        self.hidden_size = hidden_size

    def forward(self, input_tensor):
        hidden_tensor = torch.zeros(self.num_layer, 1, self.hidden_size)
        output, hidden = self.rnn(input_tensor, hidden_tensor)
        output = output[-1,:,:]
        output = self.fc(output)
        output = self.softmax(output)
        return output