from flax import linen as nn
import jax
import jax.numpy as jnp
class AgentNeuralNetwork(nn.Module):
    numhiddenlayers : int
    outputs : int
    def setup(self, layerwidth : list[int], numhiddenlayers : int):
        self.layerwidth = jnp.array(layerwidth) #no. of neurons per layer, can be different per layer
        self.numhiddenlayers = numhiddenlayers
        self.layers = jnp.array(nn.Dense)
        for width in layerwidth:#creates layers array of Dense objects for each width given
            layer = nn.Dense(width)
            self.layers.append(layer)

    @nn.compact #removes need for setup, best for small models so maybe change later?
    #below function is called when model() is called.
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)#x (input) = output of layer
            x = nn.relu(x)#x = output of activation function
        return x
    #def output(params, inp, model):
    #   output = model.apply(params, inp)
    #    return output
    
key = jax.random.PRNGKey(0) 
initinput = jnp.ones((1,4)) #nonce input for policy to be initialised.
ann = AgentNeuralNetwork(50,2) #no. of inputs, no. of neurons in hidden layer
params = ann.init(key, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)




#to instantiate: (maybe put in training.py?)
