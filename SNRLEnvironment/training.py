from Env1 import EnvCollection, State, Agent, Action, statestep, addvelocity
from renderer import drawframe, drawwindow, showplt, update
from batching import create_envbatch
import jax
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from agentneuralnetwork import train_step, optimizer

#creates key for subkeys to be made from
envnum = 40 #number of environments
episodelength = 10
objnum = 2 #num of objects in environment



def agentact(input):
    #output = jnp.clip(agent.policy(input),min=-2,max=2)
    output = agent.policy(input)
    #grads = train(logits,input,optimizer)
    action = jnp.clip(output, min=-2, max=2)
    #optimizer.update(grads)
    #output = agent.apply(input)
    #test = agent.policy.test()
    #actionval = agent(input)
    return action, output #random action

#wrap statestep and agentact in jit? will this parallelise execution of them?
pureact = jax.jit(agentact)

windowwidth = 600
windowheight = 600
# env limits must be >= than window size
limits = jnp.array([0,600]) 
#environments = EnvCollection(envstates=envstates)
#vmap the construction of the environments?

env1img = []
env1frames = []
window = drawwindow(windowheight,windowwidth)
fig = plt.figure()
@jax.jit
def runstep(currentstates):
    #agents act on environments
    agentact = jax.vmap(pureact,in_axes=(0)) #define the agentact func as applying pureact to the num of environments in a parallel way
    actions, outputs = agentact(currentstates) #apply the agentact function to the current states of all environments in parallel, output is a jnp array of shape (envnum, 2) where each row is the action for that environment
    
    #step through states, note currentstates[1] is the agent states and currentstates[0] is the goal states
    newstates = jax.vmap(statestep,in_axes=(0,0,None))(currentstates,actions,limits) #apply the agent's transformations to the states
    currentstates = jnp.array(newstates)
    #extract first env state for image
    return currentstates
@jax.jit
def train(policy,action, state, optimizer):
    loss = train_step(policy,action, state, optimizer)
    #grads = jax.vmap(train_step, in_axes=(0,0))(logits,data)
    return loss
@jax.jit
def createimages(state, window):
    frame = drawframe(state, window)
    return frame
@jax.jit
def drawframes(envstates,window):
    frames = jax.vmap(createimages,in_axes=(0,None))(envstates, window)# shape = (envnum, ...)
    return frames
#@jax.jit
#def computeloss(policy, data, objnum):
#    loss, logits = jax.vmap(loss_fun, in_axes=(None,0,None))(policy,data,objnum)
#    return loss, logits

j = 0
draw = True
episodenum = 100

agent = Agent()
while j < episodenum:
    i = 0
    seed = 1830 + j
    key = jax.random.key(seed)
    envstates = create_envbatch(key, envnum)
    while i < episodelength:
        envstates = runstep(envstates)
        #grads = jnp.mean(grads,axis=0)
        train_step(agent.policy,envstates,optimizer)
        #jax.vmap(optimizer.update)(grads)
        #optimizer.update(grads)
        data = envstates
        if draw:
            frames = drawframes(envstates,window)
            env1frame = createimages(envstates[5],window)
            env1frames.append(env1frame)
        i += 1
        print(i)
        #data = frames #for input being an image
    print("j ",j)
    j += 1
    #loss, logits = computeloss(agent.policy,data,objnum)
i = 0
for frame in env1frames:
    image = plt.imshow(frame,animated=True)
    env1img.append([image])
    i += 1
if draw:
    ani = anim.ArtistAnimation(fig, env1img, interval=2, repeat_delay=1000)
    plt.show()

    
