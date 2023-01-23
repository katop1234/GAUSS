import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class Digit:
    '''
    Instantiates a Digit object with just a number from 0->9. Can get the one_hot_encoded representation also
    '''
    def __init__(self, value):
        assert value in list(range(10))
        self.value = value

        digit = self.value
        one_hot_encode_list = [0] * 10
        one_hot_encode_list[digit] = 1
        self.one_hot_encoded = torch.tensor(one_hot_encode_list)
    
    def __str__(self):
        return "Digit(" + str(self.value) + ")"
    
    def value(self):
        return self.value
    
    def get_one_hot_encoded(self):
        return self.one_hot_encoded
    
    def get_digit_from_one_hot_encoded(one_hot_encoded_tensor):
        assert type(one_hot_encoded_tensor) is torch.Tensor
        count = 0
        for element in one_hot_encoded_tensor:
            if element == torch.Tensor([1]):
                return Digit(count)
            count += 1
        
        assert 1==0, "it should've returned a number already!"

class _N_Digit_Number:
    def __init__(self, N, number):
        assert number < 10**N
        assert type(number) is int
        self.N = N
        self.number = number

        tensor_representation = list()
        str_number = str(number).zfill(N)
        self.str_number = str_number

        for str_digit in str_number:
          int_digit = int(str_digit)
          digit = Digit(int_digit)
          one_hot_encoded_digit = digit.get_one_hot_encoded()
          tensor_representation.append(one_hot_encoded_digit)
        
        self.list_of_one_hot_encoded_digits = (tensor_representation)
    
    def get_num_digits(self):
        return self.N
    
    def number(self):
        return self.number
    
    def get_list_of_tensor_digits(self):
        output = list()
        for char in str(self):
            output.append(torch.Tensor([float(char)]))
        return output
    
    def get_list_of_one_hot_encoded_digits(self):
        return self.list_of_one_hot_encoded_digits

    def __str__(self):
        return self.str_number
    
    def get_nth_digit(self, n):
        num_digits = self.N
        last_digit_0_indexed = num_digits - 1
        assert 0 <= n < num_digits

        one_hot_encoded_digit = self.list_of_one_hot_encoded_digits[last_digit_0_indexed-n]
        return Digit.get_digit_from_one_hot_encoded(one_hot_encoded_digit)

class _8_Digit_Number(_N_Digit_Number):
    def __init__(self, number):
        super().__init__(8, number)

class AI(nn.Module):
    def __init__(self, input_size, layer1_size, output_size):
        super(AI, self).__init__()
        self.layer1 = nn.Linear(input_size, layer1_size)
        self.output = nn.Linear(layer1_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.output(x)
        return x

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

