import torch
import torch.nn as nn

class Learner:
    def __init__(self, model, learning_rate):
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    def update(self, name_tensor, category_tensor):
        output = self.model(name_tensor)
        loss = self.criterion(output, category_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

