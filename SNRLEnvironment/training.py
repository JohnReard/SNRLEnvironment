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

def agentact(input, key, subkey):
    #output = jnp.array((random.randint(-10,10),random.randint(-10,10)))
    output = jnp.array(jax.random.randint(subkey, shape=(2,), minval=-10, maxval=10))
    #actionval = agent.apply(input)
    #test = agent.policy.test()
    #actionval = agent(input)
    del key
    return output #random action

#wrap statestep and agentact in jit? will this parallelise execution of them?
pureact = jax.jit(agentact)

windowwidth = 600
windowheight = 600
# env limits must be >= than window size
envnum = 2
limits = jnp.array([-600,600]) 
environments = EnvCollection(envstates=jnp.array([initialstate1,initialstate2]), envlimits=jnp.vstack([limits,limits]), coordlimits=jnp.array([2,2]))
#vmap the construction of the environments?

currentstates = environments.envstates
frames = []
window = drawwindow(windowheight,windowwidth)
fig = plt.figure()
i=0
while i < 100:
    #retrieve random key
    key, subkey = jax.random.split(key)
    #agents act on environments
    agentact = jax.vmap(pureact,in_axes=(envnum,None, None)) #define the agentact func as applying pureact to the num of environments in a parallel way
    actions = agentact(jnp.array([currentstates]), key, subkey) #apply the agentact function to the current states of all environments in parallel, output is a jnp array of shape (envnum, 2) where each row is the action for that environment
    #step through states
    newstates = jax.vmap(statestep,in_axes=(0,0,None))(currentstates,actions,limits) #apply the agent's transformations to the states
    currentstates = jnp.array(newstates)
    #extract first env state for image
    env1states = currentstates[0]

    #create images and append to animation[to keep image of previous locations set frame = to window =.]
    frame = drawframe(env1states, window)
    image = plt.imshow(frame, animated=True)
    frames.append([image])
    i+=1
ani = anim.ArtistAnimation(fig, frames, interval=2, repeat_delay=1000)
plt.show()

    
