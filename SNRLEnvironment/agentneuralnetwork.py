
from flax import nnx as nnx
import jax
import optax #loss function from here
import jax.numpy as jnp
class AgentNeuralNetwork(nnx.Module):
    #hiddenlayers : int
    #outputs : int
    #n_features will be 2*objnum in no img input, and (windowx * 2 + windowy * 2) * 3 in img
    def __init__(self,n_features : int = 4, n_hidden: int=4,n_targets: int=2,*, rngs : nnx.Rngs):#creates layers
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features,n_hidden, rngs=rngs)   
        self.layer2 = nnx.Linear(n_features,n_hidden*10, rngs=rngs)
        self.layer3 = nnx.Linear(n_features*10,n_targets, rngs=rngs) 
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
ann = AgentNeuralNetwork(rngs = nnx.Rngs(0)) #seed neural net with rng
#optimizer = nnx.ModelAndOptimizer(ann, optax.sgd(learning_rate=0.05),wrt=nnx.Param) #uses stndrd gradient descent algo as optimiser
optimizer = nnx.Optimizer(ann, tx=optax.sgd(learning_rate=0.05),wrt=nnx.Param)
#@jax.jit
def extrgoalagentstate(state):
    return state[0],state[1]
@jax.jit
def compgrads(loss):
    grad = nnx.grad(loss)
    return grad
#@nnx.jit
#@nnx.jit
#def loss_fun(logits, idealstate : jax.Array,agentloc):
    #logits = model(data)#raw output is from applying model to data
    #predictedstate = jax.vmap(lambda x, y : x + y)(agentloc, logits)
#    print(jnp.shape(agentloc),jnp.shape(logits))
#    predictedstate = agentloc+logits
#    print(jnp.shape(idealstate))
#    nnx.value_and_grad 
#    loss = optax.squared_error(predictions =predictedstate, targets=idealstate).mean()
#    return loss #returns loss and raw output
#@nnx.jit #why jit this and not other funcs???
def train_step(policy, states, optimizer):
    print(jnp.shape(states))
    def lossfn(policy,states):
        outputs = jax.vmap(policy)(states)
        agentlocs, goallocs = jax.vmap(extrgoalagentstate)(states)
        distances = jax.vmap(lambda x,y: x - y)(goallocs,agentlocs)
        #loss = jax.vmap(optax.squared_error)(outputs,distances).mean()
        return jax.vmap(optax.squared_error)(outputs,distances).mean() #action and ideal action
    #loss gradient is one scalar sum of all individual runs' losses
    #shape is (n,objnum, 2) or (environments, object states, coordinates)
    #losses = jax.vmap(loss_fun)(outputs,idealstates,agentloc)
    #gradloss = nnx.grad(loss_fun)
    valgrad = nnx.grad(lossfn, has_aux=True)
    grads, loss = valgrad(policy,states)
    print(loss)
    #grads = loss_fun(outputs,idealstates,agentloc)
    optimizer.update(policy,grads) 
    #print(jax.tree.structure(grads))
    #print(jax.tree.structure(nnx.state(policy, nnx.Param)))
    #grads = nnx.grad(losses)
    #optimizer.update()
    #jax.vmap(optimizer.update)(grads)
    #can't call loss_gradient because logits = model(data) can't be called
    #return grads


#params = ann.init(key, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)

