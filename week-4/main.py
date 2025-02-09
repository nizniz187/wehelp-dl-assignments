from layer import *
from network import Network
from activation_function import *
from loss_function import *

print('Neural Network for Regression Tasks:')

nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [[0.8, 0.4], [-0.5, 0.5], [0.6, -0.25]], ActivationFunction.relu)
  ],
  Layer(LayerType.OUTPUT, 2)
)

inputs=[1.5, 0.5]
expects=[0.8, 1]
outputs=nn.forward(*inputs)
total_loss = LossFunction.mse(outputs, expects)
print(f'inputs={inputs}, outputs={outputs}, expects={expects}, total loss={total_loss}')

inputs=[0,1]
expects=[0.5, 0.5]
outputs=nn.forward(*inputs)
total_loss = LossFunction.mse(outputs, expects)
print(f'inputs={inputs}, outputs={outputs}, expects={expects}, total loss={total_loss}')

####################################################################################################

print()
print('Neural Network for Binary Classification Tasks:')

nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [[0.8], [0.4], [-0.5]], ActivationFunction.relu)
  ],
  Layer(LayerType.OUTPUT, 1, None, None, ActivationFunction.sigmoid)
)

inputs=[0.75, 1.25]
expects=[1]
outputs=nn.forward(*inputs)
total_loss = LossFunction.binary_cross_entropy(outputs, expects)
print(f'input=({inputs}), outputs={outputs}, expects={expects}, total loss={total_loss}')

inputs=[-1, 0.5]
expects=[0]
outputs=nn.forward(*inputs)
total_loss = LossFunction.binary_cross_entropy(outputs, expects)
print(f'input={inputs}, outputs={outputs}, expects={expects}, total loss={total_loss}')

####################################################################################################

print()
print('Neural Network for Multi-Label Classification Tasks:')

nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75], [0.6, 0.5, -0.5]], ActivationFunction.relu)
  ],
  Layer(LayerType.OUTPUT, 3, None, None, ActivationFunction.sigmoid)
)

inputs=[1.5, 0.5]
expects=[1, 0, 1]
outputs=nn.forward(*inputs)
total_loss = LossFunction.binary_cross_entropy(outputs, expects)
print(f'input=({inputs}), outputs={outputs}, expects={expects}, total loss={total_loss}')

inputs=[0, 1]
expects=[1, 1, 0]
outputs=nn.forward(*inputs)
total_loss = LossFunction.binary_cross_entropy(outputs, expects)
print(f'input={inputs}, outputs={outputs}, expects={expects}, total loss={total_loss}')

####################################################################################################

print()
print('Neural Network for Multi-Class Classification Tasks:')

nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75], [0.6, 0.5, -0.5]], ActivationFunction.relu)
  ],
  Layer(LayerType.OUTPUT, 3, None, None, ActivationFunction.softmax)
)

inputs=[1.5, 0.5]
expects=[1, 0, 0]
outputs=nn.forward(*inputs)
total_loss = LossFunction.categorical_cross_entropy(outputs, expects)
print(f'input=({inputs}), outputs={outputs}, expects={expects}, total loss={total_loss}')

inputs=[0, 1]
expects=[0, 0, 1]
outputs=nn.forward(*inputs)
total_loss = LossFunction.categorical_cross_entropy(outputs, expects)
print(f'input={inputs}, outputs={outputs}, expects={expects}, total loss={total_loss}')