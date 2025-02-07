from enum import Enum
from decimal import Decimal, getcontext

getcontext().prec = 10

class LayerType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:
  def __init__(self, type, neuronAmt, bias=None, weights=None):
    self.type = type
    self.neuronAmt = neuronAmt
    if(self.type != LayerType.OUTPUT):
      self.bias = bias
      self.weights = weights
    self.outputAmt = len(weights[0]) if weights else 0

  def calcOutputs(self, inputs):
    ##print(inputs)
    if(self.outputAmt == 0):
      return
    
    outputs = []
    for i in range(self.outputAmt):
      o = Decimal(0)
      for j in range(len(inputs)):
        o += Decimal(inputs[j]) * Decimal(self.weights[j][i])
      o += Decimal(self.bias) * Decimal(self.weights[-1][i])
      outputs.append(float(o.normalize()))
    return outputs