import jax
import jax.numpy as jnp
import optax
from flax import nnx

class AgentNeuralNetwork(nnx.Module):
    #hiddenlayers : int
    #outputs : int
    #n_features will be 2*objnum in no img input, and (windowx * 2 + windowy * 2) * 3 in img
    def __init__(self,n_features : int = 4, n_hidden: int=4,n_targets: int=2,*, rngs : nnx.Rngs):#creates layers
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features,n_hidden, rngs=rngs)   
        self.layer2 = nnx.Linear(n_features,n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_features,n_targets, rngs=rngs) 
        #optimiser adjusts parameters/weights(?) of model to minimise error returned by cost function  
    def __call__(self, x):
        #x is input
        #flatten x to make shape of input match shape of first layer (n_features)
        x = x.reshape(self.n_features) #[[agentx agenty] , [goalx goaly]]
        #pass input to layers and add activation func
        x = self.layer1(nnx.selu(x))
        x = self.layer2(nnx.selu(x))
        x = self.layer3(x)
        return x
key = jax.random.PRNGKey(100)
