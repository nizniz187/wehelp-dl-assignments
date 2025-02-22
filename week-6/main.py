import torch

import csv
from decimal import Decimal
from layer import *
from network import Network
from activation_function import *
from loss_function import *

print('Task 1: Regression Task - Predict Weight by Gender and Height')

# nn = Network(
#   Layer(LayerType.INPUT, 2, 0, Network.init_weights_he(2, 2)),
#   [
#     Layer(LayerType.HIDDEN, 2, 0, Network.init_weights_xavier(2, 1), ActivationFunction.sigmoid),
#   ],
#   Layer(LayerType.OUTPUT, 1),
#   LossFunction.mse
# )

# def encode_gender(gender):
#   gender = gender.strip().lower()
#   if gender == 'male':
#     return 1
#   elif gender == 'female':
#     return 0
#   else:
#     return -1

# inputs = []
# expects = []
# with open('gender-height-weight.csv', mode='r', newline='', encoding='utf-8') as file:
#   csv_reader = csv.DictReader(file)
#   next(csv_reader)

#   line_count = 0
#   for row in csv_reader:
#     line_count += 1
#     gender = encode_gender(row['Gender'])
#     if gender == -1:
#       continue
#     inputs.append([gender, Decimal(row['Height'])])
#     expects.append([Decimal(row['Weight'])])
#     if line_count == 5000:
#       break
    
# print('training...')
# nn.execute(inputs, expects, 0.01, 1)
# print(f'| total loss={nn.get_total_loss()}')
# print(f'| new weights={nn.get_fixed_weights()}')

print('Task 3: Regression Task - Predict Weight by Gender and Height')

list = [[2, 3, 1], [5, -2, 1]]
tensor = torch.tensor(list)
print(f'Task 3-1: shape={tensor.shape}, dtype={tensor.dtype}')

tensor = torch.rand(3, 4, 2)
print(f'Task 3-2: shape={tensor.shape}, list={tensor.tolist()}')

tensor = torch.ones(2, 1, 5)
print(f'Task 3-3: shape={tensor.shape}, list={tensor.tolist()}')

tensor = torch.tensor([[1, 2, 4], [2, 1, 3]])
tensor2 = torch.tensor([[5], [2], [1]])
result = tensor @ tensor2
print(f'Task 3-4: result={result.tolist()}')

tensor = torch.tensor([[1, 2], [2, 3], [-1, 3]])
tensor2 = torch.tensor([[5, 4], [2, 1], [1, -5]])
result = tensor * tensor2
print(f'Task 3-5: result={result.tolist()}')