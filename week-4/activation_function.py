import math

class ActivationFunction:
  @staticmethod
  def linear(x):
    return x
  
  @staticmethod
  def relu(x):
    return x if x > 0 else 0
  
  @staticmethod
  def sigmoid(x):
    return 1 / (1 + math.exp(-x))