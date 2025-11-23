from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
class AgentNeuralNetwork(nn.Module):
    numhiddenlayers : int
    layerwidth : list[int] #no. of neurons per layer, can be different per layer
    numoutputs : int #no. of neurons in output layer
    
    
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

#optimisers are used because otherwise the raw output would be too random and counterintuitive so optimisers find a gradient for all these outputs.
optimiser = optax.adam(learning_rate=0.1) #higher learning rate = more randomness
optimiserstate = optimiser.init(params) #state is updated every step

#no grads value!!



state = TrainState.create(apply_fn=ann.apply, params=params, tx=optimiser)

class Training():
    def step(state, batch):
        loss, grads = grad_fn(state, state.params, batch)
        state = state.apply_gradients(grads=grads)
        updates, state = optimiser.update(grads ,optimiserstate, params) #update optimiser each step (should be in function that is called every envstep (?))
        params = optax.apply_updates(params, updates)
        return state, loss, params
    #loss function returns value representing model accuracy
    def loss(state, params, batch): #batch???
        data_input, labels = batch
        predictions = state.apply_fn(params, data_input)
        #resize??
        loss = optax.sigmoid_binary_cross_entropy(predictions, labels).mean()
        return loss 
    grad_fn = jax.value_and_grad(loss, argnums=1) #returns loss function and gradient (argnums might be wrong?)
#output = ann.apply(params, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)




#to instantiate: (maybe put in training.py?)
