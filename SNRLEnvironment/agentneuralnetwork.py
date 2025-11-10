from flax import linen as nn
import jax
import jax.numpy as jnp
class AgentNeuralNetwork(nn.Module):
    @nn.compact #removes need for setup, best for small models so maybe change later?
    def __call__(self, hiddenlayers, outputs,):
        self.hiddenlayers : int
        self.outputs : int
        x = nn.Dense(features = self.hiddenlayers)(x)
        x = nn.Dense(features = self.outputs)(x)
        x = nn.tanh(x)
        return x
    def init(seed, input, inputdimensions): #input will be agentpos & goalpos
        model = AgentNeuralNetwork(hiddenlayers=100, outputs=4)
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed),2)
        inp = jax.random.normal(input, inputdimensions)
        params = model.init(init_rng, inp)
    def output(params, inp, model):
        output = model.apply(params, inp)
        return output
#to instantiate: (maybe put in training.py?)
