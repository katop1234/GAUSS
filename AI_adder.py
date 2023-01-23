import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
import matplotlib.pyplot as plt
from classes import AI, Digit, _N_Digit_Number, _8_Digit_Number

def generate_random_N_digit_number(N):
    num_digits = N
    '''
    Returns original integer and length N list of one-hot encoded digits.
    This represents the actual number and the neural network input representation.
    '''
    integer = random.randint(0, 10**N - 1)
    return _N_Digit_Number(N, integer)

def generate_random_8_digit_number():
    return generate_random_N_digit_number(8)

def generate_training_sample_for_N_digits(N):
    num1 = generate_random_N_digit_number(N)
    num2 = generate_random_N_digit_number(N)

    integer_sum = num1.number + num2.number

    X = num1.get_list_of_tensor_digits() + num2.get_list_of_tensor_digits()
    X = torch.cat(X)
    y_digit_rep = _N_Digit_Number(N+1, integer_sum)
    y = y_digit_rep.get_list_of_tensor_digits()
    y = torch.cat(y)

    return X, y

def generate_training_sample_for_8_digits():
    '''
    Returns a X, y pair. 
    X is a list of one-hot encoded digits (length 16)
    y is a list of one-hot encoded digits (length 9)
    '''
    return generate_training_sample_for_N_digits(8)

def pause():
    print("paused code")
    time.sleep(100000)

# Instantiate the neural network
input_size = 16
layer1_size = 256
output_size = 9
learning_rate = 1e-5
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AI(input_size, layer1_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Todo uncomment if you want to collect the data again
# Generate some dummy data for the inputs and outputs
num_samples = 400

X_train = []
y_train = []
for sample in range(num_samples):
    X, y = generate_training_sample_for_N_digits(8)
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
    if count % 100 == 0:
        print ("count", count, 'Loss: {:.4f}'.format(loss.item()))
        losses.append(loss.item())

# plt.plot(losses)
# plt.show()

# todo plot the fucntion(0+n) against identity and see how the curve looks

predictions = []
actual = []
for i in range(0, int(1e7), int(1e3)):
    actual.append(i)
    num1 = _8_Digit_Number(i)
    num2 = _8_Digit_Number(0)

    integer_sum = num1.number + num2.number

    X = num1.get_list_of_tensor_digits() + num2.get_list_of_tensor_digits()
    X = torch.cat(X)
    
    prediction = net(X)
    print(type(prediction), prediction)
    predictions.append(float(prediction))

plt.plot(actual, "r")
plt.plot(predictions, "b")
plt.show()
