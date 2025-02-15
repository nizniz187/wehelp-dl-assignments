from decimal_tool import DecimalTool as D

class ActivationFunction:
  @staticmethod
  def linear(x):
    return D.to_decimal(x)
  
  @staticmethod
  def linear_prime(x):
    return D.ONE
  
  @staticmethod
  def relu(x):
    return x if x > 0 else D.ZERO
  
  @staticmethod
  def relu_prime(x):
    return D.ONE if x > 0 else D.ZERO
  
  @staticmethod
  def sigmoid(x):
    return D.ONE / (D.ONE + D.exp(-x))
  
  @staticmethod
  def sigmoid_prime(x):
    return ActivationFunction.sigmoid(x) * (D.ONE - ActivationFunction.sigmoid(x))
  
  @staticmethod
  def softmax(inputs):
    inputs = D.to_decimal(list(inputs))
    maxInput = max(inputs)
    maxDiffSum = sum(D.exp(x - maxInput) for x in inputs)
    outputs = []
    for x in inputs:
      o = D.exp(x - maxInput) / maxDiffSum
      outputs.append(o)
    return outputs
  
  @staticmethod
  def get_prime_function(function):
    return getattr(ActivationFunction, function.__name__ + '_prime', None)