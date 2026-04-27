import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
from Env1 import  Agent, statestep, addvelocity
from renderer import drawframe, drawwindow, showplt, update
from batching import create_envbatch
import jax
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from matplotlib.animation import FFMpegWriter
import seaborn as sb
from Env1 import train_step, lossfn, act

from flax import nnx as nnx
import optax
import os 
jax.config.update("jax_enable_x64", True)
#creates key for subkeys to be made from

devices = jax.devices()
mesh = jax.make_mesh((len(devices),),('databatch'))
sharding = jax.sharding.NamedSharding(mesh, P('databatch'))

envnum = 6 #number of environments
episodelength = 200
objnum = 2 #num of objects in environment
rad = 10
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
def createimages(state, window, collision, statobjs):
    frame = drawframe(state, window, collision, statobjs)
    return frame
@jax.jit
def drawframes(envstates,window):
    frames = jax.vmap(createimages,in_axes=(0,None,None))(envstates, window)# shape = (envnum, ...)
    return frames

#fig1 = plt.figure()
def envinit(objnum, objrad, policy, optimizer, seed):
    #init
    seed = seed*seed
    featurenum = 2 * (objnum+2)
    envstates, statobjs = create_envbatch(seed, envnum,limits,objnum,objrad,3)
    #init policy
    policy = policy(rngs = nnx.Rngs(0), n_features= featurenum)
    optimizer = nnx.Optimizer(policy, tx=optimizer,wrt=nnx.Param)
    #sharding
    splitstates = jnp.split(envstates, len(devices))
    devind = 0
    for batch in splitstates:
        envstates = jax.device_put(envstates, devices[devind])
        #should append to envstates really?
        print(batch.shape)
        #shape = envstates.shape
        #jax.debug.visualize_sharding(envstates,shape)
        devind +=1
        #jax.debug.visualize_array_sharding(envstates)

    window = drawwindow(600,600)

    goalinitstates = jax.vmap(lambda stt: stt[0])(envstates)
    agentinitstates = jax.vmap(lambda stt: stt[1])(envstates)
    return policy, optimizer, envstates,statobjs, window
def envstep(episodeindex,envstates,statobjs, policy, losses):
    actions = act(policy,envstates)
    envstates, collision = runstep(envstates,actions)
    loss, outputs = lossfn(policy,envstates)
    losses.append(loss)
    return envstates, statobjs, loss, collision, losses
def drawenv(envstates, statobjs, collision, window, env1frames):
    #frames = drawframes(envstates,window)
    #firstenvcol.append(collision[0])
    print("shape is ",jnp.array(statobjs).shape)# should be 3,2 but is 30
    print(statobjs[3])
    env1frame = createimages(envstates[3],window,collision[3],statobjs[3])
    env1frames.append(env1frame)
def f(state,agentstate):
    objx = jnp.array(state[0])
    objy = jnp.array(state[1])
    objrad = jnp.array(state[2])
    return jnp.where(((objx +  objrad < agentstate[0] - agentstate[2]) & (objx - objrad < agentstate[0] - agentstate[2]))
            |((objy -  objrad > agentstate[1] + agentstate[2]) & (objy +  objrad > agentstate[0] - agentstate[2])),jnp.array([objx,objy,objrad]),)
def animate(frames):
    j = 0
    for frame in frames:  
        image = plt.imshow(frame,animated=True)
        env1img.append([image])
        print("Drawing frame ", j)
        j+=1
    #write env1img to a .txt or .json or something and then have a separate script that renders that into anims with the code below?
    ani = anim.ArtistAnimation(fig, env1img, interval=1, repeat_delay=1000)
    ani.save("animationenv.gif", writer='pillow', fps=30)

def drawplots(losses,episodelength):
    fig, plots = plt.subplots(1,3)
    plots[0].plot(jnp.arange(0,len(losses)),losses)  
    means=[]
    i=0
    while i < len(losses):
        lossgraph = losses[i-episodelength:i]
        lossgraph = jnp.array(lossgraph)
        lossgraph = jnp.mean(lossgraph)
        means.append(lossgraph)
        i+=episodelength
    plots[1].plot(jnp.arange(0,len(means)),means)
    plt.savefig("plots.png")



    #((xmin,xmax), (ymin,ymax), density) = jax.vmap(lambda stt: ((stt[0]-rad,stt[0]+rad),(stt[1]-rad,stt[1]+rad),1))(goalinitstates)
    #xs, ys = (xmin,xmax),(ymin,ymax)
    #goalinitstates = jnp.array(jax.vmap(lambda sts:[sts[0].astype(int),sts[1].astype(int)])(goalinitstates))
    #heatmap = sb.heatmap(goalinitstates,annot=False,cmap='viridis')
    #plots[2] = heatmap
    #plt.savefig("plots.png")
    #print(firstenvcol)