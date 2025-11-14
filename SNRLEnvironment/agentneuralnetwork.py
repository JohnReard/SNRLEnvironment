from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
class AgentNeuralNetwork(nn.Module):
    numhiddenlayers : int
    layerwidth : list[int] #no. of neurons per layer, can be different per layer
    numoutputs : int #no. of neurons in output layer
    #optimisers are used because otherwise the raw output would be too random and counterintuitive so optimisers find a gradient for all these outputs.
    key = jax.random.PRNGKey(5) 
    initinput = jnp.ones((1,4)) #nonce input for policy to be initialised.
    ann = AgentNeuralNetwork(layerwidth=[20,10,5], numhiddenlayers=3, numoutputs=2) #no. of inputs, no. of neurons in hidden layer
    params = ann.init(key, initinput)
    optimiser = optax.adam(learning_rate=0.1)
    initopt = optimiser.init(params)
    def setup(self):
        #self.layers = nn.ModuleList # Create an empty list
        self.layers = [nn.Dense(width) for width in self.layerwidth] #creates layers array of Dense objects for each given width (only list comprehension compiles, classic for loop does not)
        self.outputlayer = nn.Dense(self.numoutputs)
    #@nn.compact #removes need for setup, best for small models so maybe change later?
    #below function is called when model() is called.
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)#x (input) = output of layer
            x = nn.relu(x)#x = output of activation function
        x = self.outputlayer(x)
        #x = nn.relu(x)
        return x
    
key = jax.random.PRNGKey(5) 
initinput = jnp.ones((1,4)) #nonce input for policy to be initialised.
ann = AgentNeuralNetwork(layerwidth=[20,10,5], numhiddenlayers=3, numoutputs=2) #no. of inputs, no. of neurons in hidden layer
params = ann.init(key, initinput)

optimiser = optax.adam(learning_rate=0.1)
initopt = optimiser.init(params)

#output = ann.apply(params, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)




#to instantiate: (maybe put in training.py?)
