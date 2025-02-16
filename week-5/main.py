from layer import *
from network import Network
from activation_function import *
from loss_function import *

print('Task 1: Neural Network for Regression Tasks')

nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [0.8, -0.5, 0.6], ActivationFunction.relu),
    Layer(LayerType.HIDDEN, 1, 1, [[0.6, -0.3], [0.4, 0.75]], ActivationFunction.linear)
  ],
  Layer(LayerType.OUTPUT, 2),
  LossFunction.mse
)

inputs = [1.5, 0.5]
expects = [0.8, 1]

print('Task 1-1:')
nn.execute(inputs, expects, 0.01, 1)
print(f'| inputs={inputs}, outputs={nn.get_outputs()}, expects={expects}')
print(f'| total loss={nn.get_total_loss()}')
print(f'| new weights={nn.get_fixed_weights()}')


print('Task 1-2:')
nn.execute(inputs, expects, 0.01, 1000)
print(f'| inputs={inputs}, outputs={nn.get_outputs()}, expects={expects}')
print(f'| total loss={nn.get_total_loss()}')
print(f'| new weights={nn.get_fixed_weights()}')

####################################################################################################

print()
print('Task 2: Neural Network for Binary Classification Tasks:')

nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [0.8, 0.4, -0.5], ActivationFunction.relu),
  ],
  Layer(LayerType.OUTPUT, 1, 0, None, ActivationFunction.sigmoid),
  LossFunction.binary_cross_entropy
)

inputs = [0.75, 1.25]
expects = [1]

print('Task 1-1:')
nn.execute(inputs, expects, 0.1, 1)
print(f'| inputs={inputs}, outputs={nn.get_outputs()}, expects={expects}')
print(f'| total loss={nn.get_total_loss()}')
print(f'| new weights={nn.get_fixed_weights()}')


print('Task 1-2:')
nn.execute(inputs, expects, 0.1, 1000)
print(f'| inputs={inputs}, outputs={nn.get_outputs()}, expects={expects}')
print(f'| total loss={nn.get_total_loss()}')
print(f'| new weights={nn.get_fixed_weights()}')