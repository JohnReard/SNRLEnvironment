from dataclasses import dataclass
import dataclasses
import jax
import agentneuralnetwork
import random
import jax.numpy as jnp
from agentneuralnetwork import ann 


devicecount = jax.device_count()
devices = jax.devices()

def addsclrs(x, y):
    return x+y

class Action:
    velocity : jnp.array #might need to be a scalar for parallelisation
    angle : int
    cost : float
    #consider making this into a jnp.array for vectorisation?
    def __init__(self, velocityx, velocityy):
        print("Action velocity:",velocityx,velocityy)
        #velocity = int(velocity)
        self.velocity = jnp.array([velocityx,velocityy])
        self.angle = 0
        self.cost = 0
@dataclass   
class State:
   goalpos : jnp.array
   agentpos : jnp.array
   #goal : bool change to an int?
class Observation:
    pass

class Agent:
    policy : agentneuralnetwork.AgentNeuralNetwork
    velocity : jnp.array
    angle : int
    action : Action
    rng : int
    init_rng : int
    inp : int
    #velocitylimit : int
    def __init__(self, initialstate : State):
        #self.policy = Policy()
        #self.knowledgeset = []
        #construct agent policy
        self.policy = ann
        self.action = Action(1,1)#placeholder
        self.velocity = jnp.array([0,0])
        self.angle = 0
        self.agentposlist = []
        #self.velocitylimit = self.policy.velocitylimit
        #self.agentpos = jnp.array((300,300))
    @jax.jit
    def observe(self, state : State):
        pass
    @jax.jit
    def act(self, env):
        policyinput = jnp.array([env.currentstate.agentpos[0],env.currentstate.agentpos[1], env.goalpos[0], env.goalpos[1]])
        #self.agentpos not changing.
        #self.params = self.policy.init(self.init_rng, self.ijnp)
        #maybe should be in init? but will have to figure out how the ijnput will go in then.

        #output = jax.vmap(lambda x : x*100)(output)
        action = Action(output[0],output[1])
        return action
        #use output to define action
        #action = Action(1,0) #placeholder
        #self.velocity += action.velocity
        #self.agentpos = tuple(map(self.agentpos + self.velocity))
        #self.angle += action.angle
    @jax.jit
    def assigncost():
        pass

def statestep(envstate,currentaction):
    #self.agent.observe(self.currentstate)
    #currentaction = env.agent.act(env)
    newvelocity = addvelocity(envstate[1], currentaction) #envstate[1] is the agent velocity
    newstate = [envstate[0],newvelocity]
    #if envstate[1][0] == envstate[0][0]:
    #    #env.goalreached()
    #    return newstate
    #else:
    return newstate
def addvelocity(a,b):#might not be vectorisable because might not be a pure function?
        newvelocity = jax.vmap(lambda x, y : x + y)(a, b)
        return newvelocity
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class EnvCollection:
    envlimits : jnp.array #jnp.array of arrays
    envstates : jnp.array #jnp.array of arrays
    velocitylimits : jnp.array #jnp.array of ints
    #agent : Agent
    #name : str

