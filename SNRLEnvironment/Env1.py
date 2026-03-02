from dataclasses import dataclass
import dataclasses
import jax
from jax import make_jaxpr
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
    def __init__(self):
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
@jax.jit
def statestep(envstate,currentaction,limits):#limits is a tuple
    #self.agent.observe(self.currentstate)
    #currentaction = env.agent.act(env)
    #newcoords = addvelocity(envstate[1], currentaction) #envstate[1] is the agent velocity [500, 300] + [actionx, actiony]
    addedcoords = jax.vmap(lambda x, y : x + y)(envstate[1], currentaction)# add agent transformation (action) to agent pos
    newcoords = jnp.clip(addedcoords,min=0, max=limits[1]) #make sure the new agent pos isn't outside of env limits assuming x = y for limits
    newstate = jnp.array([envstate[0],newcoords])
    #might have to concatenate the agent state to goal state in another vmapped function 
    #if envstate[1][0] == envstate[0][0]:
    #    #env.goalreached()
    #    return newstate
    #else:
    return newstate
@jax.jit
def addvelocity(a,b):#might not be vectorisable because might not be a pure function?
        addedcoords = jax.vmap(lambda x, y : x + y)(a, b)
        newcoords = jnp.clip(addedcoords,min=-600, max=600) #Clips x and y coords to -limits and limits #LIMITS ARE HARDCODED, CHANGE IN FUTURE
        return newcoords
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class EnvCollection:
    #envlimits : jnp.array #jnp.array of arrays
    envstates : jnp.array #jnp.array of arrays
    #coordlimits : jnp.array #jnp.array of ints

