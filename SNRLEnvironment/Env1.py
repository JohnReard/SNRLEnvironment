import jax
import jax.numpy as jnp
import agentneuralnetwork
from agentneuralnetwork import params, ann
#from flax import linen #maybe change this model library?
class Action:
    velocity : int
    angle : int
    cost : float
    def __init__(self, velocity, angle):
        self.velocity = velocity
        self.angle = angle
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
    #knowledgeset : list[Observation]
    velocity : int
    angle : int
    action : Action
    agentpos : jnp.array
    rng : int
    init_rng : int
    ijnp : int
    #params : int
    def __init__(self, initstate : State):
        #self.policy = Policy()
        #self.knowledgeset = []

        #construct agent policy
        self.policy = ann
        self.agentpos = initstate.agentpos
        self.action = Action(1,0)#placeholder
        self.velocity = 0
        self.angle = 0
        #self.agentpos = jnp.array((300,300))
    def observe(self, state : State):
        pass
    def act(self, env):
        policyinput = jnp.array([self.agentpos, env.goalpos])

        #self.params = self.policy.init(self.init_rng, self.ijnp)
        #maybe should be in init? but will have to figure out how the ijnput will go in then.
        output = self.policy.apply(params,policyinput) #maybe change params to a field?
        print("output:",output)
        output = jnp.mean(output)
        action = Action(output,0)
        print("velocity:",self.velocity)
        #self.velocity += action.velocity
        print("output mean:",output)
        return action
        #use output to define action
        #action = Action(1,0) #placeholder
        #self.velocity += action.velocity
        #self.agentpos = tuple(map(self.agentpos + self.velocity))
        #self.angle += action.angle

    def assigncost():
        pass
    #action = jnp.where(
    #        action_magnitude > params.max_robot_speed,
    #         action * params.max_robot_speed / action_magnitude,
    #        action
    #    )

class Environment:
    limits : jnp.array #change name or functionality in future? list of vectors that define limits of 2d environment.
    actionspace : list[Action]
    currentstate : State
    agent : Agent
    def __init__(self, limits:jnp.array, initialstate : State, agent : Agent):
        self.limits = limits
        self.currentstate = initialstate
        self.goalpos = initialstate.goalpos
        self.agent = agent #agent will have to be an already initialised object
    def statestep(self):
        self.agent.observe(self.currentstate)
        currentaction = self.agent.act(self)
        self.agent.velocity = currentaction.velocity
        self.newstate = self.currentstate
        print("current agent pos:", self.currentstate.agentpos)
        if self.newstate.agentpos[0] <= self.limits[0] and self.newstate.agentpos[1] <= self.limits[1]:
            self.newstate.agentpos += self.currentstate.agentpos + self.agent.velocity
            
        #newstate = State(self.currentstate.goalpos, self.agent.agentpos)
        print("agent pos is:", self.agent.agentpos)
        self.currentstate = self.newstate
        #Maybe add newstate to knowledgeset?
        self.newstate = None
        print("agentpos:",type(self.agent.agentpos), "goalpos:", type(self.currentstate.goalpos))
        if self.agent.agentpos[0] == self.currentstate.goalpos[0]:
            self.goalreached()
        
    def statereset():
        pass
    def episodeend(self):
        pass
        #self.agent.assigncost()
        #self.statereset()
    def goalreached(self):
        self.episodeend()
