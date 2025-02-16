from decimal_tool import DecimalTool as D
from loss_function import LossFunction
from layer import LayerType

class Network:
  def __init__(self, input_layer, hidden_layer, output_layer, loss_function, learning_rate = 0.01):
    self.layers = [input_layer, *hidden_layer, output_layer]
    self.loss_function = loss_function
    self.loss_function_prime = LossFunction.get_prime_function(loss_function)
    self.learning_rate = D(learning_rate)

  def execute(self, inputs, expects, loop_count):
    self.expects = expects

    for _ in range(loop_count):
      if getattr(self, 'fixed_weights', None):
        self.set_weights(self.fixed_weights)
      self.weights = D.to_decimal(self.get_weights())
      self.outputs = self.forward(*inputs)
      self.total_loss = self.calc_total_loss(self.outputs, self.expects)
      self.gradients = self.backward(self.expects, self.weights)
      self.fixed_weights = self.zero_grad(self.weights, self.gradients, self.learning_rate)
      # print([float(fw) for fw in self.fixed_weights], self.get_weights())

  def forward(self, *inputs):
    layer_inputs = list(inputs)
    for layer in self.layers:
      layer_outputs = layer.execute(layer_inputs)
      # print(layer_outputs)
      layer_inputs = layer_outputs
    return layer_outputs
    
  def calc_total_loss(self, outputs, expects):
    return self.__execute_loss_function(outputs, expects)
  
  def backward(self, expects, weights):
    gradients = list(weights)
    self.neuron_gradients = [None] * sum(l.neuron_size for l in self.layers)
    for i in reversed(range(len(weights))):
      layer_index = self.get_layer_index_by_weight_index(i)
      layer_weight_index = self.get_layer_weight_index(i)
      layer = self.layers[layer_index]
      layer_output_index = layer.get_output_index_by_weight_index(layer_weight_index)
      input = layer.get_input_by_weight_index(layer_weight_index)
      
      gradients[i] = self.__calc_gradient(layer_index + 1, layer_output_index, expects) * input # Output -> Weight
      # print(i, layer_index, layer_weight_index, layer_output_index, input)
    # print(self.neuron_gradients)
    return gradients
    
  def zero_grad(self, weights, gradients, learning_rate):
    fixed_weights = []
    for i in range(len(weights)):
      fixed_weights.append(weights[i] - learning_rate * gradients[i])
    return fixed_weights
    # fixed_weights = list(self.weights)
    # fixed_weights[0] = weights[0] - learning_rate * gradients[0]
    # fixed_weights[-1] = weights[-1] - learning_rate * gradients[-1]
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
  
  def get_layer_index_by_weight_index(self, index):
    weight_count = 0
    for i in range(len(self.layers)):
      layer = self.layers[i]
      weight_count += layer.get_weight_size()
      # print(i, weight_count, index)
      if weight_count > index:
        return i
      
  def get_layer_weight_index(self, index):
    weight_index = index
    for i in range(len(self.layers)):
      layer = self.layers[i]
      weight_size = layer.get_weight_size()
      if weight_index < weight_size:
        return weight_index
      else:
        weight_index -= weight_size

  def get_neuron_index(self, layer_index, neuron_index):
    return sum(self.layers[i].neuron_size for i in range(layer_index)) + neuron_index

  def set_weights(self, weights):
    index = 0
    for l in self.layers:
      weight_size = l.get_weight_size()
      l.set_weights(weights[index:index + weight_size])
      # print(index, index + weight_size, weights[index:index + weight_size])
      index += weight_size

  def __calc_gradient(self, layer_index, neuron_index, expects):
    network_neuron_index = self.get_neuron_index(layer_index, neuron_index)
    # print(network_neuron_index, self.neuron_gradients[network_neuron_index])
    if self.neuron_gradients[network_neuron_index]:
      return self.neuron_gradients[network_neuron_index]

    layer = self.layers[layer_index]
    gradient = 0
    if layer.type == LayerType.OUTPUT:
      gradient = self.__execute_loss_function_prime(layer.outputs[neuron_index], expects[neuron_index]) # Loss -> Output
      # print(layer_index, neuron_index, gradient)
    else:
      for ni in range(layer.output_size):
        g = self.__calc_gradient(layer_index + 1, ni, expects) # Loss -> Next Input
        g *= layer.get_weight(ni) # Next Input -> Output
        gradient += g
        # print(layer_index, neuron_index, ni, gradient)
    gradient *= layer.execute_activation_function_prime(layer.inputs[neuron_index]) # Output -> Input
    self.neuron_gradients[network_neuron_index] = gradient
    return gradient
  
    # gradient1 = self.__execute_loss_function_prime(outputs[0], expects[0])
    # gradient1 *= self.layers[-1].execute_activation_function_prime(self.layers[-1].inputs[0])
    # gradient1 *= self.layers[-2].get_weight(0)
    # gradient2 = self.__execute_loss_function_prime(outputs[-1], expects[-1])
    # gradient2 *= self.layers[-1].execute_activation_function_prime(self.layers[-1].inputs[-1])
    # gradient2 *= self.layers[-2].get_weight(1)
    # gradient = gradient1 + gradient2
    # gradient *= self.layers[-2].execute_activation_function_prime(self.layers[-2].inputs[0])
    # gradient *= self.layers[0].inputs[0]
    # gradients[0] = gradient
    
    # gradient = self.__execute_loss_function_prime(outputs[-1], expects[-1])
    # gradient *= self.layers[-1].execute_activation_function_prime(self.layers[-1].inputs[-1])
    # gradient *= self.layers[-2].outputs[-1]
    # gradients[-1] = gradient
  
  def __execute_loss_function(self, outputs, expects):
    return self.loss_function(outputs, expects)
  
  def __execute_loss_function_prime(self, output, expect):
    if(self.loss_function_prime is LossFunction.mse_prime):
      return self.loss_function_prime(output, expect, len(self.outputs))
    else:
      return self.loss_function_prime(output, expect)