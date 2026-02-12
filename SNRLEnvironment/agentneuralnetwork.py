
from flax import nnx as nnx
import jax
import jax.numpy as jnp
class AgentNeuralNetwork(nnx.Module):
    #hiddenlayers : int
    #outputs : int
    def __init__(self, rngs): ## DUMMY PLACEHOLDER, implement later
        self.linear1 = nnx.Linear(2,3, rngs=rngs)   
        self.linear2 = nnx.Linear(2,3, rngs=rngs)   
    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x
    def test(self):
        pass
        
    #def __call__(self, x):
    #    self.hiddenlayers : int
    #    self.outputs : int
    #    x = nnx.linear(x)
    #    x = nnx.tanh(x) #what does dropout do? what do these functions in call() do?
    #    x = nnx.Dense(self.outputs)(x)
    #    return x
    #def output(params, inp, model):
    #   output = model.apply(params, inp)
    #    return output

#CREATE A BLOCK AND MODEL CLASS
    
#key = jax.random.PRNGKey(0) PRNG key does not work on GPU, maybe try jax.random instead?
initinput = jnp.ones((1,4)) #nonce input for policy to be initialised.
ann = AgentNeuralNetwork(rngs = nnx.Rngs(0)) #seed neural net with rng
#params = ann.init(key, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)

