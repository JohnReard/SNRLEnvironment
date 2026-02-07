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

