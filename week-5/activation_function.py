from decimal import Decimal, getcontext

class ActivationFunction:
  @staticmethod
  def linear(x):
    return x
  
  @staticmethod
  def linear_prime(x):
    return Decimal(1)
  
  @staticmethod
  def relu(x):
    return x if x > 0 else Decimal(0)
  
  @staticmethod
  def relu_prime(x):
    return Decimal(1) if x > 0 else Decimal(0)
  
  @staticmethod
  def sigmoid(x):
    return Decimal(1) / (Decimal(1) + Decimal.exp(-x))
  
  @staticmethod
  def sigmoid_prime(x):
    return ActivationFunction.sigmoid(x) * (Decimal(1) - ActivationFunction.sigmoid(x))
  
  @staticmethod
  def softmax(inputs):
    maxInput = max(inputs)
    maxDiffSum = sum(Decimal.exp(x - maxInput) for x in inputs)
    outputs = []
    for x in inputs:
      o = Decimal.exp(x - maxInput) / maxDiffSum
      outputs.append(o)
    return outputs