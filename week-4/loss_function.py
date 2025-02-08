import math

class LossFunction:
  @staticmethod
  def mse(outputs, expects):
    outputAmt = len(outputs)
    return 1 / outputAmt * sum((expects[i] - outputs[i]) ** 2 for i in range(outputAmt))
  
  @staticmethod
  def binary_cross_entropy(outputs, expects):
    #print(outputs, expects)
    outputAmt = len(outputs)
    return -(sum((expects[i] * math.log(outputs[i]) + (1 - expects[i]) * math.log(1 - outputs[i])) for i in range(outputAmt)))