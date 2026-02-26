
from flax import nnx as nnx
import jax
import optax #loss function from here
import jax.numpy as jnp
class AgentNeuralNetwork(nnx.Module):
    #hiddenlayers : int
    #outputs : int
    
    def __init__(self,n_features : int = 64, n_hidden: int=100,n_targets: int=1,*, rngs : nnx.Rngs):#creates layers
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features,n_hidden, rngs=rngs)   
        self.layer2 = nnx.Linear(n_features,n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_features,n_hidden, rngs=rngs) 
        #optimiser adjusts parameters/weights(?) of model to minimise error returned by cost function  
    def __call__(self, x):
        #x is input
        #flatten x??
        #pass input to layers and add activation func
        x = self.layer1(nnx.selu(x))
        x = self.layer2(nnx.selu(x))
        x = self.layer3(x)
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
optimizer = nnx.ModelAndOptimizer(ann, optax.sgd(learning_rate=0.05)) #uses stndrd gradient descent algo as optimiser
def loss_fun(model : nnx.Module, data : jax.Array, goalpos : jax.Array):
    logits = model(data)#raw output is from applying model to data
    loss = optax.squared_error(predictions =logits, targets=goalpos)
    return loss, logits #returns loss and raw output
@nnx.jit #why jit this and not other funcs???
def train_step(model : nnx.Module, data : jax.Array, optimizer : nnx.Optimizer):
    loss_gradient = nnx.grad(loss_fun, has_aux=True)#gradient of loss returned by loss func
    grads, logits = loss_gradient(model, data) #logits is the raw predictions
    optimizer.update(grads)


#params = ann.init(key, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)

