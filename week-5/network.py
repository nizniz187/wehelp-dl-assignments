from decimal_tool import DecimalTool as D
from loss_function import LossFunction

class Network:
  def __init__(self, input_layer, hidden_layer, output_layer, loss_function, learning_rate = 0.01):
    self.layers = [input_layer, *hidden_layer, output_layer]
    self.loss_function = loss_function
    self.loss_function_prime = LossFunction.get_prime_function(loss_function)
    self.learning_rate = D(learning_rate)

  def execute(self, inputs, expects, loop_count):
    self.expects = expects

    for _ in range(loop_count):
      self.reset()
      self.outputs = self.forward(*inputs)
      self.total_loss = self.calc_total_loss(self.outputs, self.expects)
      self.gradients = self.backward(self.expects, self.total_loss, self.weights)
      self.fixed_weights = self.zero_grad(self.weights, self.gradients, self.learning_rate)
      self.set_weights(self.fixed_weights)

  def reset(self):
    self.outputs = self.total_loss = self.gradients = None
    self.weights = D.to_decimal(self.get_weights())

  def forward(self, *inputs):
    layer_inputs = list(inputs)
    for layer in self.layers:
      layer_outputs = layer.execute(layer_inputs)
      # print(layer_outputs)
      layer_inputs = layer_outputs
    return layer_outputs
    
  def calc_total_loss(self, outputs, expects):
    return self.loss_function(outputs, expects)
  
  def backward(self, expects, total_loss, weights):
    gradients = list(weights)
    
    gradient = self.loss_function_prime(self.outputs[-1], expects[-1], len(self.outputs))
    gradient += self.layers[-1].activation_function_prime(self.layers[-1].inputs[-1])
    gradient += self.layers[-2].outputs[-1]
    gradients[-1] = gradient
    return gradients
    
  def zero_grad(self, weights, gradients, learning_rate):
    # fixed_weights = []
    # for i in range(len(weights)):
    #   fixed_weights.append(weights[i] - learning_rate * gradients[i])
    # return fixed_weights
    fixed_weights = list(self.weights)
    fixed_weights[-1] = weights[-1] - learning_rate * gradients[-1]
    return fixed_weights
  
  def get_outputs(self):
    return [float(o) for o in self.outputs]
  
  def get_total_loss(self):
    return float(self.total_loss)
  
  def get_fixed_weights(self):
    return [float(w) for w in self.fixed_weights]
  
  def get_weights(self):
    weights = []
    for l in self.layers:
      weights += [float(w) for w in l.get_weights()]
    return weights
  
  def set_weights(self, weights):
    index = 0
    for l in self.layers:
      weight_size = l.get_weight_size()
      l.set_weights(weights[index:index + weight_size])
      index += weight_size