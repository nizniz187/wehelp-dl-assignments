from itertools import islice
import math
import torch
import torch.nn as nn
import csv

print('Task 1: Regression Task - Predict Weight by Gender and Height')
class WeightPredictionModel(nn.Module):
  def __init__(self):
    super(WeightPredictionModel, self).__init__()
    self.fc1 = nn.Linear(2, 64) # input -> hidden layer 1
    self.fc2 = nn.Linear(64, 32) # hidden layer 1 -> 2
    self.fc3 = nn.Linear(32, 1) # hidden layer 2 -> output
    
  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def encode_gender(gender):
  gender = gender.strip().lower()
  if(gender == 'female'):
    return 0
  elif(gender == 'male'):
    return 1
  else:
    return -1

model = WeightPredictionModel() # create nn model
criterion = nn.MSELoss()  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.09)

# prepare training data
inputs = []
expects = []
with open('gender-height-weight.csv', mode='r', newline='', encoding='utf-8') as file:
  csv_reader = csv.DictReader(file)
  next(csv_reader)

  line_count = 0
  for row in csv_reader:
    line_count += 1
    gender = encode_gender(row['Gender'])
    if gender == -1:
      continue
    inputs.append([gender, float(row['Height'])])
    expects.append([float(row['Weight'])])
    if line_count == 5000:
      break

train_data = torch.tensor(inputs)
train_labels = torch.tensor(expects)

# training
loss_sum = 0
epochs = 10000
for epoch in range(epochs):
  # forward
  outputs = model(train_data)
  loss = criterion(outputs, train_labels)
  # backward
  optimizer.zero_grad()  # clear last gradient
  loss.backward()
  optimizer.step() # adjust weights

  if((epoch + 1) % 1000 == 0):
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# prepare evaluation data
inputs = []
expects = []
with open('gender-height-weight.csv', mode='r', newline='', encoding='utf-8') as file:
  csv_reader = csv.DictReader(file)
  next(csv_reader)

  line_count = 5001
  for row in islice(csv_reader, line_count, None):
    line_count += 1
    gender = encode_gender(row['Gender'])
    if gender == -1:
      continue
    inputs.append([gender, float(row['Height'])])
    expects.append([float(row['Weight'])])

train_data = torch.tensor(inputs)
train_labels = torch.tensor(expects)

# evaluation
outputs = model(train_data)
loss = criterion(outputs, train_labels)

print(f"Average Loss: {loss.item():.4f}, Average Error: {math.sqrt(loss.item()):.2f} pounds")

#######################################################################

print('')
print('Task 2: Binary Classification Task - Predict Survival Status of Passengers on Titanic')
class TitanicSurvivalModel(nn.Module):
  def __init__(self):
    super(TitanicSurvivalModel, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(6, 16),
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU(),
      nn.Linear(8, 1),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.model(x)

model = TitanicSurvivalModel() # create nn model
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# prepare training data
inputs = []
expects = []
with open('titanic.csv', mode='r', newline='', encoding='utf-8') as file:
  csv_reader = csv.DictReader(file)
  next(csv_reader)

  line_count = 0
  for row in csv_reader:
    line_count += 1

    pclass = int(row['Pclass'])
    gender = encode_gender(row['Sex'])
    age = float(row['Age']) if row['Age'] else 0
    sibsp = int(row['SibSp'])
    parch = int(row['Parch'])
    fare = float(row['Fare'])
    survived = float(row['Survived'])

    if gender == -1 or age == 0:
      continue
    inputs.append([pclass, gender, age, sibsp, parch, fare])
    expects.append([survived])
    if line_count == 500:
      break

train_data = torch.tensor(inputs)
train_labels = torch.tensor(expects)

# training
loss_sum = 0
epochs = 10000
for epoch in range(epochs):
  # forward
  outputs = model(train_data)
  loss = criterion(outputs, train_labels)
  # backward
  optimizer.zero_grad()  # clear last gradient
  loss.backward()
  optimizer.step() # adjust weights

  if((epoch + 1) % 1000 == 0):
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# prepare evaluation data
inputs = []
expects = []
with open('titanic.csv', mode='r', newline='', encoding='utf-8') as file:
  csv_reader = csv.DictReader(file)
  next(csv_reader)

  line_count = 501
  for row in islice(csv_reader, line_count, None):
    line_count += 1
    
    pclass = int(row['Pclass'])
    gender = encode_gender(row['Sex'])
    age = float(row['Age']) if row['Age'] else 0
    sibsp = int(row['SibSp'])
    parch = int(row['Parch'])
    fare = float(row['Fare'])
    survived = float(row['Survived'])

    if gender == -1 or age == 0:
      continue
    inputs.append([pclass, gender, age, sibsp, parch, fare])
    expects.append([survived])

train_data = torch.tensor(inputs)
train_labels = torch.tensor(expects)

# evaluation
outputs = model(train_data)
loss = criterion(outputs, train_labels)

y_true = torch.tensor(expects)  # 真實標籤 (0 or 1)
y_pred_labels = (outputs >= 0.5).float()
accuracy = (y_pred_labels == y_true).float().mean()

print(f"Average Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")

#######################################################################

print('')
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