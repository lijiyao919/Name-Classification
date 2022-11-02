import torch.nn as nn

class LSTM(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, num_layers, n_category):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, n_category)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        output, hidden = self.lstm(input_tensor)
        output = output[-1,:,:]
        output = self.fc(output)
        output = self.softmax(output)
        return output