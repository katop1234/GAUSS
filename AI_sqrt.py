import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import torch, torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from classes import AI, _N_Digit_Number, _8_Digit_Number, Digit, MyDataset

class AI(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, output_size, dropout=0.1):
        super(AI, self).__init__()
        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output = nn.Linear(layer2_size, output_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x

# Task at hand
function = lambda a, b: int(str(a + b)[-1])
upper_bound = 9

# Instantiate the neural network
input_size = 2
layer1_size = 32
layer2_size = 16
output_size = 1
learning_rate = 1e-2
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AI(input_size, layer1_size, layer2_size, output_size, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

num_samples = 200
batch_size = 2

X_train = []
y_train = []
for sample in range(num_samples):
    X1 = random.randint(0, upper_bound)
    X2 = random.randint(0, upper_bound)
    y = function(X1, X2)
    X = torch.tensor([X1, X2], dtype=torch.float32)
    y = torch.tensor([y], dtype=torch.float32)
    X_train.append(X)
    y_train.append(y)

# Create an instance of the custom dataset
dataset = MyDataset(X_train, y_train)

# Create a data loader to handle batching of data
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses = []
# Now you can use the data loader to iterate over the training data
for count, (X_batch, y_batch) in enumerate(dataloader):
    # Do something with the data, e.g. pass it to the neural network
    outputs = net(X_batch)
    loss = criterion(outputs, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % 10 == 0:
        print ("count", count, 'Loss: {:.4f}'.format(loss.item()))
    losses.append(loss.item())

print("PREDICTIONS")   
for _ in range(20):
    x1 = random.randint(0, upper_bound)
    x2 = random.randint(0, upper_bound)
    test_input = torch.tensor([x1, x2], dtype=torch.float32)
    prediction = net(test_input)

    print("x1 x2", x1, x2, "prediction", float(prediction), "actual", function(x1, x2))

print("PREDICTIONS") 

plt.plot(losses)
plt.show()

exit()


predictions_curve = []
sqrts = []

for i in range(0, upper_bound):
    test_input = torch.tensor([i], dtype=torch.float32)
    prediction = net(test_input)
    predictions_curve.append(float(prediction))

    sqrts.append(function(i)) 

# plt.plot(predictions_curve, "r")
# plt.plot(sqrts, "b")
# plt.show()
