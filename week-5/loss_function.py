from decimal import Decimal, getcontext

class LossFunction:
  @staticmethod
  def mse(outputs, expects):
    output_size = len(outputs)
    # print(outputs, expects)
    return Decimal(1) / Decimal(output_size) * sum((Decimal(expects[i]) - Decimal(outputs[i])) ** 2 for i in range(output_size))
  
  @staticmethod
  def mse_prime(output_size, output, expect):
    # print(outputs, expects)
    return Decimal(2) / Decimal(output_size) * (Decimal(output) - Decimal(expect))
  
  @staticmethod
  def binary_cross_entropy(outputs, expects):
    # print(outputs, expects)
    return -(sum((Decimal(expects[i]) * Decimal.ln(Decimal(outputs[i])) + (Decimal(1) - Decimal(expects[i])) * Decimal.ln(Decimal(1) - Decimal(outputs[i]))) for i in range(len(outputs))))
  
  @staticmethod
  def binary_cross_entropy_prime(output, expect):
    # print(outputs, expects)
    return -(Decimal(expect) / Decimal(output)) + ((Decimal(1) - Decimal(expect)) + (Decimal(1) - Decimal(output)))

  @staticmethod
  def categorical_cross_entropy(outputs, expects):
    return -(sum((Decimal(expects[i]) * Decimal.ln(outputs[i])) for i in range(len(outputs))))