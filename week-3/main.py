from layer import *
from network import Network

print('Neural Network 1:')
nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [[0.8], [0.4], [-0.5]])
  ],
  Layer(LayerType.OUTPUT, 1)
)
outputs=nn.forward(1.5, 0.5)
print(f'input=(1.5, 0.5), outputs={outputs}')
outputs=nn.forward(0, 1)
print(f'input=(0, 1), outputs={outputs}')

print('Neural Network 2:')
nn = Network(
  Layer(LayerType.INPUT, 2, 1, [[0.5, 0.6], [1.5, -0.8], [0.3, 1.25]]),
  [
    Layer(LayerType.HIDDEN, 2, 1, [[0.6], [-0.8], [0.3]]),
    Layer(LayerType.HIDDEN, 1, 1, [[0.5, -0.4], [0.2, 0.5]])
  ],
  Layer(LayerType.OUTPUT, 2)
)
outputs=nn.forward(0.75, 1.25)
print(f'input=(0.75, 1.25), outputs={outputs}')
outputs=nn.forward(-1, 0.5)
print(f'input=(-1, 0.5), outputs={outputs}')