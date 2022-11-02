import torch
import matplotlib.pyplot as plt
from learner import Learner
from rnn import RNN
from lstm import LSTM
from gru import GRU
from utils import ALL_LETTERS, N_LETTERS
from utils import letter_to_tensor, name_to_tensor, retrieve_train_test_set

class Trainer:
    def __init__(self, model, learning_rate, train_set, test_set, category, name):
        self.model = model
        self.learner = Learner(model, learning_rate)
        self.n_iters = 20
        self.all_losses = []
        self.all_accuracy = []
        self.train_set, self.test_set, self.category = train_set, test_set, category
        self.name = name

    def train(self):
        print(f"Training the {self.name}:")
        for i in range(self.n_iters):
            current_loss = 0
            for name_tensor, category_tensor in self.train_set:
                output, loss = self.learner.update(name_tensor, category_tensor)
                current_loss += loss
            aver_loss = current_loss / len(self.train_set)
            test_correct_rate = self.test()
            self.all_losses.append(aver_loss)
            self.all_accuracy.append(test_correct_rate)
            print(f"{i + 1}  {(i + 1) / self.n_iters * 100}  {aver_loss:.2f}  {test_correct_rate:.2f}%")
        print("\n")

    def test(self):
        num_correct = 0
        num_samples = len(self.test_set)

        for name_tensor, category_tensor in self.test_set:
            output = self.model(name_tensor)
            _, pred = torch.max(output, dim=1)
            num_correct += bool(pred == category_tensor.item())

        return num_correct / num_samples * 100



    def predict(self, input_line):
        print(f"\n> {input_line}")
        with torch.no_grad():
            name_tensor = name_to_tensor(input_line)
            output = model(name_tensor)
            _, pred = torch.max(output, dim=1)
            print(self.category[pred.item()]+"\n")


train_set, test_set, category = retrieve_train_test_set()
n_categories = len(category)

# RNN
learning_rate = 0.005
n_hidden = 128
n_layer = 2
model = RNN(N_LETTERS, n_hidden, n_layer, n_categories)
rnn_trainer = Trainer(model, learning_rate, train_set, test_set, category, "RNN")
rnn_trainer.train()

#LSTM
learning_rate = 0.01
n_hidden = 128
n_layer = 2
model = LSTM(N_LETTERS, n_hidden, n_layer, n_categories)
lstm_trainer = Trainer(model, learning_rate, train_set, test_set, category, "LSTM")
lstm_trainer.train()

#GRU
learning_rate = 0.01
n_hidden = 128
n_layer = 2
model = GRU(N_LETTERS, n_hidden, n_layer, n_categories)
gru_trainer = Trainer(model, learning_rate, train_set, test_set, category, "GRU")
gru_trainer.train()

#plot figure
plt.figure(0)
plt.plot(rnn_trainer.all_losses, label="RNN", marker="o")
plt.plot(lstm_trainer.all_losses, label="LSTM", marker="^")
plt.plot(gru_trainer.all_losses, label="GRU", marker="*")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.show()

plt.figure(1)
plt.plot(rnn_trainer.all_accuracy, label="RNN", marker="o")
plt.plot(lstm_trainer.all_accuracy, label="LSTM", marker="^")
plt.plot(gru_trainer.all_accuracy, label="GRU", marker="*")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Accuracy (%)")
plt.show()


#self test
print("please select a triner:\n")
print("1. RNN Trainer:\n")
print("2. LSTM Trainer:\n")
print("3. GRU Trainer:\n")
num = int(input("Type the number:"))
if num==1:
    trainer = rnn_trainer
elif num==2:
    trainer = lstm_trainer
elif num==3:
    trainer = gru_trainer
else:
    raise Exception("No such trainer.")

while True:
    sentence = input("Input a name:")
    if sentence == "quit":
        break
    trainer.predict(sentence)