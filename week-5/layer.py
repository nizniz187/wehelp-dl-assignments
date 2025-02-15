from enum import Enum
from decimal_tool import DecimalTool as D
from activation_function import *

class LayerType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:
  def __init__(self, type, neuron_size, bias=0, weights=None, activation_function=ActivationFunction.linear):
    self.type = type
    self.neuron_size = neuron_size
    self.activation_function = activation_function
    self.activation_function_prime = ActivationFunction.get_prime_function(activation_function)
    self.bias = D.to_decimal(bias)
    self.weights = D.to_decimal(weights)
    self.output_size = self.get_output_size(weights, neuron_size)

  def execute(self, inputs):
    #print(inputs)
    if(self.output_size == 0):
      return
    
    self.inputs = [D.to_decimal(i) for i in inputs]
    activated_inputs = self.execute_activation_function(inputs)
    if(self.type == LayerType.OUTPUT):
      self.outputs = activated_inputs
    else:
      self.outputs = []
      for i in range(self.output_size):
        o = D.to_decimal(0)
        for j in range(len(self.inputs)):
          o += activated_inputs[j] * (self.weights[j][i] if isinstance(self.weights[j], list) else self.weights[j])
          #print(o, activated_inputs[j], self.weights[j][i])
        o += self.bias * (self.weights[-1][i] if isinstance(self.weights[-1], list) else self.weights[-1])
        #print(o)
        self.outputs.append(o.normalize())
      #print(self.outputs)
    return self.outputs
  
  def execute_activation_function(self, inputs):
    if(self.activation_function is ActivationFunction.softmax):
      return self.activation_function(inputs)
    else:
      outputs = []
      for i in inputs:
        outputs.append(self.activation_function(i))
      return outputs
    
  def get_output_size(self, weights, neuron_size):
    if isinstance(weights, list):
      return len(weights[0]) if isinstance(weights[0], list) else 1
    else:
      return neuron_size

  def get_weights(self):
    if self.type == LayerType.OUTPUT:
      return []
    elif isinstance(self.weights[0], list):
      weights = []
      for l in self.weights:
        weights += l
      return weights
    else:
      return self.weights
    
  def get_weight_size(self):
    if self.type == LayerType.OUTPUT:
      return 0
    else:
      return len(self.weights) * (len(self.weights[0]) if isinstance(self.weights[0], list) else 1)
  
  def set_weights(self, weights):
    if self.type == LayerType.OUTPUT:
      return
    else:
      index = 0
      for i in range(len(self.weights)):
        if(isinstance(self.weights[i], list)):
          for j in range(len(self.weights[i])):
            self.weights[i][j] = weights[index]
        else:
          self.weights[i] = weights[index]
        index += 1