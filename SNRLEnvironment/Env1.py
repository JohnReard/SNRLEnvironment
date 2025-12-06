import jax
import agentneuralnetwork
import random
import jax.numpy as jnp
from agentneuralnetwork import params, ann
#from flax import linen #maybe change this model library?
devicecount = jax.device_count()
devices = jax.devices()

def addsclrs(x, y):
    return x+y

class Action:
    velocity : jnp.array #might need to be a scalar for parallelisation
    angle : int
    cost : float
    def __init__(self, velocityx, velocityy):
        print("Action velocity:",velocityx,velocityy)
        #velocity = int(velocity)
        self.velocity = jnp.array([velocityx,velocityy])
        self.angle = 0
        self.cost = 0   
class State:
   goalpos : jnp.array
   agentpos : jnp.array
   goal : bool 
   def __init__(self, goalpos, agentpos):
         self.goalpos = jnp.array(goalpos)
         self.goal = False
         self.agentpos = jnp.array(agentpos)
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
    def __init__(self, initstate : State):
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
    def observe(self, state : State):
        pass
    def act(self, env):
        policyinput = jnp.array([env.currentstate.agentpos[0],env.currentstate.agentpos[1], env.goalpos[0], env.goalpos[1]])
        self.agentposlist.append(env.currentstate.agentpos)
        #self.agentpos not changing.
        #self.params = self.policy.init(self.init_rng, self.ijnp)
        #maybe should be in init? but will have to figure out how the ijnput will go in then.
        output = jnp.array((random.randint(1,10),random.randint(1,10))) #maybe change params to a field?
        print("output:", output)
        output = jax.vmap(lambda x : x*100)(output)
        action = Action(output[0],output[1])
        return action
        #use output to define action
        #action = Action(1,0) #placeholder
        #self.velocity += action.velocity
        #self.agentpos = tuple(map(self.agentpos + self.velocity))
        #self.angle += action.angle

    def assigncost():
        pass
def statestep(env):
    #self.agent.observe(self.currentstate)
    currentaction = env.agent.act(env)
    newvelocity = addvelocity(env, currentaction)
    newstate = State(env.currentstate.goalpos,newvelocity) 
    if env.currentstate.agentpos[0] == env.currentstate.goalpos[0]:
        env.goalreached()
        env.currentstate = newstate
    else:
        print("newstate:",newstate.agentpos)
        env.currentstate = newstate
def addvelocity(environment,currentaction):#might not be vectorisable because might not be a pure function?
        newvelocity = jax.vmap(lambda x, y : x + y)(environment.agent.velocity, currentaction.velocity)
        return newvelocity
class Environment:
    limits : jnp.array #change name or functionality in future? list of vectors that define limits of 2d environment.
    actionspace : list[Action]
    currentstate : State
    velocitylimit : int
    agent : Agent
    agentposlist = []
    agenvelocitylist = []
    
    def __init__(self, limits:jnp.array, initialstate : State, agent : Agent,velocitylimit : int):
        self.limits = limits
        self.currentstate = initialstate
        self.goalpos = initialstate.goalpos
        self.agent = agent #agent will have to be an already initialised 
        self.velocitylimit = velocitylimit
    def statereset():
        pass
    def episodeend(self):
        pass
        #self.agent.assigncost()
        #self.statereset()
    def goalreached(self):
        self.episodeend()
