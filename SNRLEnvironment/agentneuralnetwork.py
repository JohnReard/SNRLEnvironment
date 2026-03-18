
from flax import nnx as nnx
import jax
import optax #loss function from here
import jax.numpy as jnp
class AgentNeuralNetwork(nnx.Module):
    def __init__(self,n_features : int,n_targets: int=2,*, rngs : nnx.Rngs):#creates layers
        self.n_features = int(n_features)
        self.n_hidden = int((n_features/2))
        self.n_targets = 2
        self.layer1 = nnx.Linear(self.n_features-14,self.n_hidden, rngs=rngs)   
        self.layer2 = nnx.Linear(self.n_hidden,self.n_hidden, rngs=rngs)
        self.layer4 = nnx.Linear(self.n_hidden,n_features,rngs=rngs)
        self.layer5 = nnx.Linear(self.n_features,self.n_targets, rngs=rngs) 
        #optimiser adjusts parameters/weights(?) of model to minimise error returned by cost function  
    def __call__(self, x):
        #x is input, list of distances to objects
        #flatten x to make shape of input match shape of first layer (n_features)
        #normalise() is too slow here
        #find distances for all objects in a vmap?
        print("shape in call: ",x.shape)
        #objstates = jnp.array(x[2:len(x)])

        #gdistx = (x[0][0] - x[1][0])/600
        #gdisty = (x[0][1] - x[1][1])/600
        distx = jax.vmap(lambda objstate,agentstate : ((objstate[0] - agentstate[0]) + (objstate[2] + agentstate[2]))/600, in_axes=(0,None))(x,x[1])
        disty = jax.vmap(lambda objstate, agentstate : ((objstate[1] - agentstate[1]) + (objstate[2] + agentstate[2]))/600, in_axes=(0,None))(x,x[1])
        #print(objdists.shape) #objdists = [[x1,][y]]
        #jax.debug.print("x shape: {objdists}", objdists=objdists)
        #goaldist = jnp.array([gdistx,gdisty])#1,2, 1*[x, y
        #objdists = jnp.array([distx,disty])
        #objdists is 1,2,3
        a = distx[0]
        b = disty[0]
        distx.at[0].set(a*100)
        disty.at[0].set(b*100)
        x = jnp.ravel(jnp.array([distx,disty]))

        #sqrt for x ??
        x1 = self.layer1(nnx.relu(x))
        x2 = self.layer2(nnx.relu(x1))
        x3 = self.layer4(nnx.relu(x2))
        x4 = self.layer5(x3)
        return nnx.tanh(x4)  
    def test(self):
        pass
        
#seed neural net with rng
#optimizer = nnx.ModelAndOptimizer(ann, optax.sgd(learning_rate=0.05),wrt=nnx.Param) #uses stndrd gradient descent algo as optimiser
@nnx.jit
def lossfn(policy,states):
    limits = (0,600)
    goallocs, agentlocs = jax.vmap(extrgoalagentstate)(states)
    print("dist shape: ",goallocs.shape)
    print("shape in loss_fn: ", states.shape)
    outputs = act(policy,states)
    xdistances = jax.vmap(lambda x,y: x[0] - y[0]) (goallocs,agentlocs)
    ydistances = jax.vmap(lambda x,y: x[1] - y[1]) (goallocs,agentlocs)
    distances = jnp.reshape(jnp.array([xdistances,ydistances]),(len(xdistances),2))  
    distances = normalise(distances,limits)
        #reward = difference between what should happen and what did happen.
        # reward = optimal action - actual action
        #errors = jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances)
        #print(jnp.shape(collision))
        #lossx = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[0],collision))
        #lossy = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[1],collision))
        #loss = jnp.mean(lossx,lossy)
    loss = jnp.mean(jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances))
    return loss, outputs #n length vector of losses for each env
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
def train_step(policy, states, optimizer, collision):
    #states = jnp.array(jax.vmap(lambda states: [states[0:1]])(fullstates))
    print(states.shape)
    @nnx.jit
    def lossfn(policy,states):
        limits = (0,600)
        goallocs, agentlocs = jax.vmap(extrgoalagentstate)(states)
        print("dist shape: ",goallocs.shape)
        print("shape in loss_fn: ", states.shape)
        outputs = act(policy,states)
        xdistances = jax.vmap(lambda x,y: x[0] - y[0]) (goallocs,agentlocs)
        ydistances = jax.vmap(lambda x,y: x[1] - y[1]) (goallocs,agentlocs)
        distances = jnp.reshape(jnp.array([xdistances,ydistances]),(len(xdistances),2))
        
        distances = normalise(distances,limits)
        #reward = difference between what should happen and what did happen.
        # reward = optimal action - actual action
        #errors = jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances)
        #print(jnp.shape(collision))
        #lossx = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[0],collision))
        #lossy = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[1],collision))
        #loss = jnp.mean(lossx,lossy)
        loss = jnp.mean(jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances))
        return loss, outputs #n length vector of losses for each env

    valgrad = nnx.value_and_grad(lossfn,has_aux=True)
    
    (loss,actions),grads = valgrad(policy,states)
    optimizer.update(policy,grads)
    return loss, actions
    
    


#params = ann.init(key, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)

