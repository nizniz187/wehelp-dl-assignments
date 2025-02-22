from decimal_tool import DecimalTool as D

class LossFunction:
  @staticmethod
  def mse(outputs, expects):
    output_size = len(outputs)
    # print(outputs, expects)
    return D.ONE / D.to_decimal(output_size) * sum((D.to_decimal(expects[i]) - D.to_decimal(outputs[i])) ** 2 for i in range(output_size))
  
  @staticmethod
  def mse_prime(output, expect, output_size):
    # print(outputs, expects)
    return D.TWO / D.to_decimal(output_size) * (D.to_decimal(output) - D.to_decimal(expect))
  
  @staticmethod
  def binary_cross_entropy(outputs, expects):
    outputs = D.to_decimal(list(outputs))
    expects = D.to_decimal(list(expects))
    # print(outputs, expects)
    return -(sum((expects[i] * D.ln(outputs[i]) + (D.ONE - expects[i]) * D.ln(D.ONE - outputs[i])) for i in range(len(outputs))))
  
  @staticmethod
  def binary_cross_entropy_prime(output, expect):
    output = D.to_decimal(output)
    expect = D.to_decimal(expect)
    # print(outputs, expects)
    return -(expect / output) + ((D.ONE - expect) + (D.ONE - output))

  @staticmethod
  def categorical_cross_entropy(outputs, expects):
    return -(sum((D.to_decimal(expects[i]) * D.ln(outputs[i])) for i in range(len(outputs))))
  
  @staticmethod
  def get_prime_function(function):
    return getattr(LossFunction, function.__name__ + '_prime', None)