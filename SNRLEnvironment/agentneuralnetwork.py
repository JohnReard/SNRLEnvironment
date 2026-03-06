
from flax import nnx as nnx
import jax
import optax #loss function from here
import jax.numpy as jnp
class AgentNeuralNetwork(nnx.Module):
    def __init__(self,n_features : int = 2, n_hidden: int=4,n_targets: int=2,*, rngs : nnx.Rngs):#creates layers
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features,12, rngs=rngs)   
        self.layer2 = nnx.Linear(12,12, rngs=rngs)
        self.layer4 = nnx.Linear(12,4,rngs=rngs)
        self.layer5 = nnx.Linear(4,n_targets, rngs=rngs) 
        #optimiser adjusts parameters/weights(?) of model to minimise error returned by cost function  
    def __call__(self, x):
        #x is input
        #flatten x to make shape of input match shape of first layer (n_features)
        #normalise() is too slow here
        distx = (x[0][0] - x[1][0])/600
        disty = (x[0][1] - x[1][1])/600
        x = jnp.ravel(jnp.array([distx,disty]))

        #sqrt for x ??
        x1 = self.layer1(nnx.relu(x))
        x2 = self.layer2(nnx.relu(x1))
        x3 = self.layer4(nnx.relu(x2))
        x4 = self.layer5(x3)
        return nnx.tanh(x4)  
    def test(self):
        pass
        
ann = AgentNeuralNetwork(rngs = nnx.Rngs(0)) #seed neural net with rng
#optimizer = nnx.ModelAndOptimizer(ann, optax.sgd(learning_rate=0.05),wrt=nnx.Param) #uses stndrd gradient descent algo as optimiser
optimizer = nnx.Optimizer(ann, tx=optax.adam(learning_rate=0.0008),wrt=nnx.Param)
@jax.jit
def extrgoalagentstate(state):
    return state[0],state[1]
@jax.jit
def normalise(arg, limits):
    maxlim = limits[1]
    return jax.vmap(lambda arg : arg / maxlim)(arg)
@jax.jit
def act(policy,states):
    return jax.vmap(policy)(states)
@nnx.jit
def train_step(policy, states, optimizer):
    @nnx.jit
    def lossfn(policy,states):
        limits = (0,600)
        outputs = act(policy,states)
        goallocs, agentlocs = jax.vmap(extrgoalagentstate)(states)
        distances = jax.vmap(lambda x,y: x - y)(goallocs,agentlocs)
        distances = normalise(distances,limits)
        #reward = difference between what should happen and what did happen.
        # reward = optimal action - actual action
        loss = jnp.mean(jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances))
        return loss, outputs #n length vector of losses for each env

    valgrad = nnx.value_and_grad(lossfn,has_aux=True)
    (loss,actions),grads = valgrad(policy,states)
    optimizer.update(policy,grads)
    return loss, actions
    
    


#params = ann.init(key, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)

