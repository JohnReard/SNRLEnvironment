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
episodelength = 600
objnum = 2 #num of objects in environment

windowwidth = 600
windowheight = 600
# env limits must be >= than window size
limits = jnp.array([0,600]) 


env1img = []
env1frames = []
#window = drawwindow(windowheight,windowwidth)
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
episodenum = 5
avglosslist = []
losses= []
actionlist = []
allanims=[]
agent = Agent()
#fig1 = plt.figure()
while j < episodenum:
    i = 0
    seed = 1962 * (j + 1)
    key = jax.random.key(seed)
    envstates = create_envbatch(key, envnum,limits)
    while i < episodelength:
        window = drawwindow(windowheight,windowwidth)
        loss,actions = train_step(agent.policy,envstates,optimizer)
        envstates = runstep(envstates,actions)
        print("loss: ", loss)
        losses.append(loss)
        if draw and j == episodenum -1:
            #frames = drawframes(envstates,window)
            env1frame = createimages(envstates[0],window)
            env1frames.append(env1frame)
        i += 1
    #allanims.append(env1frames)
    j += 1
print("len env1frames: ",len(env1frames))
print("len allanims: ", len(allanims))
env1img = []
j=1
#while j < len(allanims):
#    i=0
#    frames = allanims[0]
#    print("len frames: ",len(frames))
#    env1img=[]
#    while i+1 < len(frames)/episodenum:
#        print(i)
for frame in env1frames:
    
    image = plt.imshow(frame,animated=True)
    #stops it running out of memory
    env1img.append([image])
    #    print("i is: ", i)
    #    i += 1
    print("j is: ", j)

    #fig = plt.figure()
    j+=1
#write env1img to a .txt or .json or something and then have a separate script that renders that into anims with the code below?

ani = anim.ArtistAnimation(fig, env1img, interval=1, repeat_delay=1000)
ani.save("animation{j}.gif", writer='pillow', fps=30)
#if draw:
#    ani = anim.ArtistAnimation(fig, env1img, interval=2, repeat_delay=1000)
#    ani.save('animation.gif', writer='pillow', fps=30)
    #plt.show()
    #ani.save('animation.gif', writer='pillow', fps=30)
fig, plots = plt.subplots(1,2)
plots[0].plot(jnp.arange(0,len(losses)),losses)  
means=[]
i=0
while i < len(losses):
    mylossthing = losses[i-episodelength:i]
    mylossthing = jnp.array(mylossthing)
    mylossthing = jnp.mean(mylossthing)
    means.append(mylossthing)
    i+=episodelength
plots[1].plot(jnp.arange(0,len(means)),means)  
plt.savefig("plots.png")