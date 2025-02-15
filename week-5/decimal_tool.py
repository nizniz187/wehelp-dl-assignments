from decimal import Decimal

class DecimalTool(Decimal):
  ZERO = Decimal(0)
  ONE = Decimal(1)
  TWO = Decimal(2)

  @staticmethod
  def to_decimal(input):
    if isinstance(input, list):  # 如果是列表，則遞歸處理
      return [DecimalTool.to_decimal(i) for i in input]
    elif input is None or not isinstance(input, (int, float)):
      return input
    else:  # 否則轉換為 Decimal
      return Decimal(input)