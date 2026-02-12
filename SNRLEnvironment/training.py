from Env1 import EnvCollection, State, Agent, Action, statestep, addvelocity
from renderer import drawframe, drawwindow, showplt, update
import jax
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim

seed = 1491
key = jax.random.key(seed) #creates key for subkeys to be made from

initialstate1 = jnp.array([[500,300],[400,300]]) #try state as a dataclass, then as just a jnp.array
initialstate2 = jnp.array([[100,200],[300,400]])
agent = Agent(initialstate1)
#input should be policyinput = jnp.array([env.currentstate.agentpos[0],env.currentstate.agentpos[1], env.goalpos[0], env.goalpos[1]])
def agentact(input, key, subkey):
    #output = jnp.array((random.randint(-10,10),random.randint(-10,10)))

    
    output = jnp.array(jax.random.randint(subkey, shape=(2,), minval=-10, maxval=10))
    #actionval = agent.apply(input)
    test = agent.policy.test()
    #actionval = agent(input)
    del key
    return output #random action


pureact = jax.jit(agentact)



windowwidth = 600
windowheight = 600
# env limits must be >= than window size
envnum = 2
limits = (windowwidth*64,windowheight*64) 
environments = EnvCollection(envstates=jnp.array([initialstate1,initialstate2]), envlimits=jnp.vstack([limits,limits]), coordlimits=jnp.array([2,2]))
#vmap the construction of the environments?

print("Environments are: ", environments)
running = True
i=0
actionset = []

currentstates = environments.envstates # this is [[goalstate1 agentstate1],[goalstate2, agentstate2]]
#window = drawwindow(windowwidth,windowheight)
#window2 = drawwindow(windowwidth,windowheight)
animationframes = []
window = drawwindow(windowheight,windowwidth)
fig = plt.figure()
i=0
while running:

    #currentstates structure: [ [ [goalx,goaly],[agentx,agenty] ] , [...] , ... ]
    
   
    key, subkey = jax.random.split(key)
    agentact = jax.vmap(pureact,in_axes=(envnum,None, None)) #define the agentact func as applying pureact to the num of environments in a parallel way
    actions = agentact(jnp.array([currentstates]), key, subkey) #apply the agentact function to the current states of all environments in parallel, output is a jnp array of shape (envnum, 2) where each row is the action for that environment
    #print("actions are: ", actions)
    #print("\nactions:", actions,"\ncurrentstates: ", currentstates)
    #actionset.append(currentstates)
    newstates, newcoords = jax.vmap(statestep)(currentstates,actions)#PROBLEM IS HERE
    currentstates = jnp.array(newstates)
    env1states = currentstates[0]
    #to keep image of previous locations set frame = to window =.
    frame = drawframe(env1states, window)
    image = plt.imshow(frame, animated=True)
    animationframes.append([image])

    #drawframe(env2,agent, window2)
    i+=1
    if i > 100:
        #print("agentposlist:", env.agentposlist)
        #print("agenvelocitylist:", env.agenvelocitylist)
        #print("agentposlist from agent:", agent.agentposlist)
        #print("actionset:", actionset)
        #print("currentstates shape: ", currentstates)
        ani = anim.ArtistAnimation(fig, animationframes, interval=2, repeat_delay=1000)
        #animation = anim.FuncAnimation(fig, update,fargs=(animationframes,0), interval = 50, repeat_delay=1000)
        plt.show()

        running = False
    
