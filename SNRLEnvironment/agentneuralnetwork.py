from flax import linen as nn
import jax
import jax.numpy as jnp
class AgentNeuralNetwork(nn.Module):
    hiddenlayers : int
    #velocitylimit : int
    outputs : int
    #def __init__(self, hiddenlayers, outputs, velocitylimit):
    #    self.hiddenlayers = hiddenlayers
    #    self.outputs = outputs
    #    self.velocitylimit = velocitylimit
        
    @nn.compact #removes need for setup, best for small models so maybe change later?
    def __call__(self, x):
        self.hiddenlayers : int
        self.outputs : int
        x = nn.Dense(self.hiddenlayers)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.outputs)(x)
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
