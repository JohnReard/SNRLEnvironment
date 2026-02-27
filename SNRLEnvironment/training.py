from Env1 import EnvCollection, State, Agent, Action, statestep, addvelocity
from renderer import drawframe, drawwindow, showplt, update
from batching import create_envbatch
import jax
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from agentneuralnetwork import loss_fun, train_step, optimizer

seed = 1486
key = jax.random.key(seed) #creates key for subkeys to be made from
envnum = 40 #number of environments
episodelength = 100


envstates, idealstates = create_envbatch(key, envnum)
agent = Agent(envstates[0])
def agentact(input,idealstate):
    inp = jnp.array([input,idealstate])
    output = agent.policy(inp)
    #grads, output = train_step(policy,input,optimizer,idealstate)
    #optimizer.update(grads)
    #output = agent.apply(input)
    #test = agent.policy.test()
    #actionval = agent(input)
    return output #random action

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
#@jax.jit
def runstep(currentstates,idealstates):
    #agents act on environments
    agentact = jax.vmap(pureact,in_axes=(0)) #define the agentact func as applying pureact to the num of environments in a parallel way
    actions = agentact(currentstates,idealstates) #apply the agentact function to the current states of all environments in parallel, output is a jnp array of shape (envnum, 2) where each row is the action for that environment
    print(actions)
    #step through states, note currentstates[1] is the agent states and currentstates[0] is the goal states
    newstates = jax.vmap(statestep,in_axes=(0,0,None))(currentstates,actions,limits) #apply the agent's transformations to the states
    currentstates = jnp.array(newstates)
    #extract first env state for image
    return currentstates
@jax.jit
def train(policy, data, optimizer ):
    jax.vmap(train_step, in_axes=(None,0,0))(policy,data,optimizer)
@jax.jit
def createimages(state, window):
    frame = drawframe(state, window)
    return frame
@jax.jit
def drawframes(envstates,window):
    frames = jax.vmap(createimages,in_axes=(0,None))(envstates, window)# shape = (envnum, ...)
    return frames
@jax.jit
def computeloss(policy, data, idealstates):
    loss, logits = jax.vmap(loss_fun, in_axes=(None,0,0))(policy,data,idealstates)
    return loss, logits
i = 0
draw = True
while i < episodelength:
    envstates = runstep(envstates,idealstates)
    data = envstates
    if draw:
        frames = drawframes(envstates,window)
        print(jnp.shape(frames))
        env1frame = createimages(envstates[0],window)
        env1frames.append(env1frame)
        #data = frames #for input being an image
    loss, logits = computeloss(agent.policy,data,idealstates)
    print("loss is: ", loss)
    i += 1

i = 0
for frame in env1frames:
    image = plt.imshow(frame,animated=True)
    env1img.append([image])
    i += 1
if draw:
    ani = anim.ArtistAnimation(fig, env1img, interval=2, repeat_delay=1000)
    plt.show()

    
