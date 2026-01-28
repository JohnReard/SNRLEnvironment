from Env1 import EnvCollection, State, Agent, Action, statestep, addvelocity
from renderer import drawframe, drawwindow
import jax
import jax.numpy as jnp
import random
initialstate = jnp.array([[500,300],[400,300]]) #try state as a dataclass, then as just a jnp.array
#print(initialstate)
agent = Agent(initialstate)
#input should be policyinput = jnp.array([env.currentstate.agentpos[0],env.currentstate.agentpos[1], env.goalpos[0], env.goalpos[1]])
def agentact(input):
    output = jnp.array((random.randint(1,10)*100,random.randint(1,10)*100))
    #actionval = agent.apply(input)
    test = agent.policy.test()
    #actionval = agent(input)
    return output #random action


pureact = jax.jit(agentact)



windowwidth = 600
windowheight = 600
#action1 = Action(0.1,0)
#action2 = Action(0,1)
#actionset = [action1,action2]
# env limits must be >= than window size
envnum = 2
limits = (windowwidth*64,windowheight*64)
#env = Environment(envlimits= jnp.array(limits),currentstate = initialstate,velocitylimit=2)
#env2 = Environment(limits= jnp.array(limits),currentstate = initialstate,velocitylimit=2) 
environments = EnvCollection(envstates=jnp.array([initialstate,initialstate]), envlimits=jnp.vstack([limits,limits]), velocitylimits=jnp.array([2,2]))
#vmap the construction of the environments?

print("Environments are: ", environments)
running = True
i=0
actionset = []

currentstates = environments.envstates

window = drawwindow(windowwidth,windowheight)
window2 = drawwindow(windowwidth,windowheight)
while running:
    
    agentact = jax.vmap(pureact,(envnum))
    actions = agentact(jnp.array([currentstates])) # test that you can call the test func
    print("\nactions:", actions,"\ncurrentstates: ", currentstates)
    actionset.append(currentstates)
    #print("currentstates: ",currentstates.dtype,"actions: ",actions.dtype)
    newstates = jax.vmap(statestep)(currentstates,actions) #add actions as arguments
    print("newstates:", newstates)
    currentstates = jnp.array(newstates)
    #drawframe(env,agent, window)
    #drawframe(env2,agent, window2)
    i+=1
    if i > 200:
        #print("agentposlist:", env.agentposlist)
        #print("agenvelocitylist:", env.agenvelocitylist)
        #print("agentposlist from agent:", agent.agentposlist)
        #print("actionset:", actionset)
        running = False
    