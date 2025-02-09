from enum import Enum
from decimal import Decimal, getcontext
from activation_function import *

getcontext().prec = 10

class LayerType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:
  def __init__(self, type, neuronAmt, bias=0, weights=None, activationFunction=ActivationFunction.linear):
    self.type = type
    self.neuronAmt = neuronAmt
    self.activationFunction = activationFunction
    self.bias = bias
    self.weights = weights
    self.outputAmt = len(weights[0]) if weights else neuronAmt

  def calcOutputs(self, inputs):
    #print(inputs)
    if(self.outputAmt == 0):
      return
    
    activatedInputs = self.executeActivationFunction(inputs)
    if(self.type == LayerType.OUTPUT):
      return activatedInputs
    else:
      outputs = []
      for i in range(self.outputAmt):
        o = Decimal(0)
        for j in range(len(inputs)):
          o += Decimal(activatedInputs[j]) * Decimal(self.weights[j][i])
          #print(o, activatedInputs[j], self.weights[j][i])
        o += Decimal(self.bias) * Decimal(self.weights[-1][i])
        #print(o)
        outputs.append(float(o.normalize()))
      #print(outputs)
      return outputs
  
  def executeActivationFunction(self, inputs):
    if(self.activationFunction is ActivationFunction.softmax):
      return self.activationFunction(inputs)
    else:
      outputs = []
      for i in inputs:
        outputs.append(self.activationFunction(i))
      return outputs