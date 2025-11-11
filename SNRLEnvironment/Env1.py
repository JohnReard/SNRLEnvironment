import jax
import jax.numpy as np
import agentneuralnetwork
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
   goalpos : np.array
   agentpos : np.array
   goal : bool 
   def __init__(self, goalpos, agentpos):
         self.goalpos = np.array(goalpos)
         self.goal = False
         self.agentpos = np.array(agentpos)
class Actionset:
    actionset : list[Action]
class Observation:
    pass

class Agent:
    policy : agentneuralnetwork.AgentNeuralNetwork
    #knowledgeset : list[Observation]
    velocity : int
    angle : int
    action : Action
    agentpos : np.array
    rng : int
    init_rng : int
    inp : int
    params : int
    def __init__(self):
        #self.policy = Policy()
        #self.knowledgeset = []

        #construct agent policy
        self.policy = agentneuralnetwork.AgentNeuralNetwork(100,2)

        self.action = Action(1,0)#placeholder
        self.velocity = 0
        self.angle = 0
        self.agentpos = (0,0)
    def observe(self, state : State):
        pass
    def act(self, env):
        policyinput = np.array([self.agentpos, env.goalpos])

        #self.params = self.policy.init(self.init_rng, self.inp)
        #maybe should be in init? but will have to figure out how the input will go in then.
        output = self.policy.model(policyinput)

        #use output to define action
        action = Action(1,0) #placeholder
        self.velocity += action.velocity
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
    limits : np.array #change name or functionality in future? list of vectors that define limits of 2d environment.
    actionspace : list[Action]
    currentstate : State
    agent : Agent
    def __init__(self, limits:np.array, actionspace : list[Action], initialstate : State, agent : Agent):
        self.limits = limits
        self.actionspace = actionspace
        self.currentstate = initialstate
        self.goalpos = initialstate.goalpos
        self.agent = agent #agent will have to be an already initialised object
    def statestep(self):
        self.agent.observe(self.currentstate)
        self.agent.act(self)
        newstate = currentstate
        newstate.agentpos += currentstate.agentpos + self.agent.velocity
        #newstate = State(self.currentstate.goalpos, self.agent.agentpos)
        print("agent pos is:", self.agent.agentpos)
        currentstate = newstate
        #Maybe add newstate to knowledgeset?
        newstate = None
        if self.agent.agentpos == currentstate.goalpos:
            self.goalreached()
        
    def statereset():
        pass
    def episodeend(self):
        agent.assigncost()
        self.statereset()
    def goalreached(self):
        self.episodeend()
