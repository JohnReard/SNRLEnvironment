from Env1 import  Agent, statestep, addvelocity
from renderer import drawframe, drawwindow, showplt, update
from batching import create_envbatch
import jax
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from matplotlib.animation import FFMpegWriter
from Env1 import train_step, lossfn, act
from agentneuralnetwork import AgentNeuralNetwork
from flax import nnx as nnx
import optax
import os
jax.config.update("jax_enable_x64", True)
#creates key for subkeys to be made from
envnum = 60000 #number of environments
episodelength = 100
objnum = 18 #num of objects in environment

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
    newstates, collision = jax.vmap(statestep,in_axes=(0,0,None))(currentstates,actions,limits) #apply the agent's transformations to the states
    #jax.debug.print("agentloc : {shape}", shape=newstates[0][1])
    currentstates = jnp.array(newstates)
    #extract first env state for image
    return currentstates, collision
@jax.jit
def createimages(state, window, collision):
    frame = drawframe(state, window, collision)
    return frame
@jax.jit
def drawframes(envstates,window):
    frames = jax.vmap(createimages,in_axes=(0,None,None))(envstates, window)# shape = (envnum, ...)
    return frames
j = 0
draw = True
episodenum = 80
avglosslist = []
losses= []
actionlist = []
allanims=[]
ann = AgentNeuralNetwork(rngs = nnx.Rngs(0),n_features= ((objnum *3)) )
optimizer = nnx.Optimizer(ann, tx=optax.adam(learning_rate=0.08),wrt=nnx.Param)
agent = Agent(policy=ann)
firstenvcol=[]
collision = 0
#fig1 = plt.figure()
while j < episodenum:
    i = 0
    seed = 6870 * (j + 1)
    key = jax.random.key(seed)
    envstates = create_envbatch(key, envnum,limits,objnum,10)
    while i < episodelength:
        window = drawwindow(windowheight,windowwidth)
        #loss,actions = train_step(agent.policy,envstates,optimizer,collision)
        actions = act(agent.policy,envstates)
        envstates, collision = runstep(envstates,actions)
        loss, outputs = lossfn(agent.policy,envstates)
        losses.append(loss)
        #if collision[0] != 0:
        #    print("collision: ", collision)
        print("loss: ", loss)
        losses.append(loss)
        if draw and j == episodenum -1:
            #frames = drawframes(envstates,window)
            firstenvcol.append(collision[0])
            env1frame = createimages(envstates[0],window,collision[0])
            env1frames.append(env1frame)
        i += 1
    #allanims.append(env1frames)
    loss, actions = train_step(agent.policy,envstates,optimizer,collision)
    episodelength += 100
    j += 1
env1img = []
j=1
@jax.jit
def f(state,agentstate):
    objx = jnp.array(state[0])
    objy = jnp.array(state[1])
    objrad = jnp.array(state[2])
    return jnp.where(((objx +  objrad < agentstate[0] - agentstate[2]) & (objx - objrad < agentstate[0] - agentstate[2]))
            |((objy -  objrad > agentstate[1] + agentstate[2]) & (objy +  objrad > agentstate[0] - agentstate[2])),jnp.array([objx,objy,objrad]),)

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
ani.save("animationenv.gif", writer='pillow', fps=30)
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
print(firstenvcol)