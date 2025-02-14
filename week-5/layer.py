from enum import Enum
from decimal import Decimal
from activation_function import *

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

  def calc_outputs(self, inputs):
    #print(inputs)
    if(self.outputAmt == 0):
      return
    
    activatedInputs = self.execute_activation_function(inputs)
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
        outputs.append(o.normalize())
      #print(outputs)
      return outputs
  
  def execute_activation_function(self, inputs):
    if(self.activationFunction is ActivationFunction.softmax):
      return self.activationFunction(inputs)
    else:
      outputs = []
      for i in inputs:
        outputs.append(self.activationFunction(i))
      return outputs
    
  def zero_grad():
    return #weight - learning_rate * gradient