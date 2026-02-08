from Env1 import EnvCollection, State, Agent, Action, statestep, addvelocity
from renderer import drawframe, drawwindow, showplt, update
import jax
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim

seed = 1001
key = jax.random.key(seed)

initialstate = jnp.array([[500,300],[400,300]]) #try state as a dataclass, then as just a jnp.array
agent = Agent(initialstate)
#input should be policyinput = jnp.array([env.currentstate.agentpos[0],env.currentstate.agentpos[1], env.goalpos[0], env.goalpos[1]])
def agentact(input):
    output = jnp.array((random.randint(-10,10),random.randint(-10,10)))
    #actionval = agent.apply(input)
    test = agent.policy.test()
    #actionval = agent(input)
    return output #random action


pureact = jax.jit(agentact)



windowwidth = 600
windowheight = 600
# env limits must be >= than window size
envnum = 2
limits = (windowwidth*64,windowheight*64) 
environments = EnvCollection(envstates=jnp.array([initialstate,initialstate]), envlimits=jnp.vstack([limits,limits]), coordlimits=jnp.array([2,2]))
#vmap the construction of the environments?

print("Environments are: ", environments)
running = True
i=0
actionset = []

currentstates = environments.envstates

#window = drawwindow(windowwidth,windowheight)
#window2 = drawwindow(windowwidth,windowheight)
animationframes = []
window = drawwindow(windowheight,windowwidth)
fig = plt.figure()
i=0
while running:

    #currentstates structure: [ [ [goalx,goaly],[agentx,agenty] ] , [...] , ... ]
    
   

    agentact = jax.vmap(pureact,(envnum))#define the agentact func as applying pureact to the num of environments in a parallel way
    actions = agentact(jnp.array([currentstates]))
    
    #print("\nactions:", actions,"\ncurrentstates: ", currentstates)
    #actionset.append(currentstates)

    newstates = jax.vmap(statestep)(currentstates,actions)
    currentstates = jnp.array(newstates)
    print("drawn states: ", currentstates[0][0],currentstates[0][1])

    window = drawframe(currentstates[0][0],currentstates[0][1], window)
    i+=1
    #image = ax.imshow(window,animated=True)
    #axis = plt.axes(limits)
    frame = plt.imshow(window, animated=True)
    animationframes.append([frame])

    #drawframe(env2,agent, window2)
    i+=1
    if i > 200:
        #print("agentposlist:", env.agentposlist)
        #print("agenvelocitylist:", env.agenvelocitylist)
        #print("agentposlist from agent:", agent.agentposlist)
        #print("actionset:", actionset)
        
        ani = anim.ArtistAnimation(fig, animationframes, interval=2, repeat_delay=1000)
        #animation = anim.FuncAnimation(fig, update,fargs=(animationframes,0), interval = 50, repeat_delay=1000)
        plt.show()

        running = False
    