from Env1 import EnvCollection, State, Agent, Action, statestep, addvelocity
from renderer import drawframe, drawwindow, showplt, update
from batching import create_envbatch
import jax
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from matplotlib.animation import FFMpegWriter
from agentneuralnetwork import train_step, optimizer
jax.config.update("jax_enable_x64", True)
#creates key for subkeys to be made from
envnum = 350 #number of environments
episodelength = 500
objnum = 2 #num of objects in environment

windowwidth = 600
windowheight = 600
# env limits must be >= than window size
limits = jnp.array([0,600]) 


env1img = []
env1frames = []
window = drawwindow(windowheight,windowwidth)
fig = plt.figure()
@jax.jit
def runstep(currentstates,actions):
    #step through states, note currentstates[1] is the agent states and currentstates[0] is the goal states
    newstates = jax.vmap(statestep,in_axes=(0,0,None))(currentstates,actions,limits) #apply the agent's transformations to the states
    print("shape of newstates ", jnp.shape(newstates) )
    currentstates = jnp.array(newstates)
    #extract first env state for image
    return currentstates
@jax.jit
def createimages(state, window):
    frame = drawframe(state, window)
    return frame
@jax.jit
def drawframes(envstates,window):
    frames = jax.vmap(createimages,in_axes=(0,None))(envstates, window)# shape = (envnum, ...)
    return frames
j = 0
draw = True
episodenum = 1
avglosslist = []
losses= []
actionlist = []
agent = Agent()
while j < episodenum:
    i = 0
    seed = 1962 * (j + 1)
    key = jax.random.key(seed)
    envstates = create_envbatch(key, envnum,limits)
    while i < episodelength:
        loss,actions = train_step(agent.policy,envstates,optimizer)
        envstates = runstep(envstates,actions)
        print("loss: ", loss)
        losses.append(loss)
        if draw:
            if j > episodenum-2:
                frames = drawframes(envstates,window)
                env1frame = createimages(envstates[0],window)
                env1frames.append(env1frame)
        i += 1
    j += 1
i = 0
for frame in env1frames:
    image = plt.imshow(frame,animated=True)
    env1img.append([image])
    i += 1
if draw:
    fig1 = plt.figure()
    ani = anim.ArtistAnimation(fig1, env1img, interval=1, repeat_delay=1000)
    ani.save('animation.gif', writer='pillow', fps=30)
fig, plots = plt.subplots(1,2)
plots[0].plot(jnp.arange(0,len(losses)),losses)  
means=[]
i=0
while i < len(losses):
    mylossthing = losses[i-500:i]
    mylossthing = jnp.array(mylossthing)
    mylossthing = jnp.mean(mylossthing)
    means.append(mylossthing)
    i+=500
plots[1].plot(jnp.arange(0,len(means)),means)  
plt.savefig("plots.png")