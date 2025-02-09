import math
from decimal import Decimal, getcontext

getcontext().prec = 10

class ActivationFunction:
  @staticmethod
  def linear(x):
    return x
  
  @staticmethod
  def relu(x):
    return x if x > 0 else 0
  
  @staticmethod
  def sigmoid(x):
    return float(Decimal(1) / (Decimal(1) + Decimal(math.exp(-x))))
  
  @staticmethod
  def softmax(inputs):
    maxInput = max(inputs)
    maxDiffSum = sum(Decimal(math.exp(x - maxInput)) for x in inputs)
    outputs = []
    for x in inputs:
      o = Decimal(math.exp(x - float(maxDiffSum))) / maxDiffSum
      outputs.append(float(o))
    return outputs