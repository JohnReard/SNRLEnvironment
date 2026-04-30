import training
from training import envstep,envinit,drawenv,animate,drawplots
from Env1 import train_step
from flax import nnx as nnx
import jax.numpy as jnp
import optax
#import your own agent implementation
from agentneuralnetwork import AgentNeuralNetwork

optimizer = optax.adam(learning_rate=0.08)


env1frames = []
losses = []
goallog = []
episodelength = 2600
episodenum = 15

envnum = 60
seed = 610
sfmcounter = 100
objnum = 10
objrad = 10

#note, featurenum is the number of objects (including agent and goal) * 3
inits, window, featurenum = envinit(objnum ,objrad, seed, envnum)

policy = AgentNeuralNetwork(rngs = nnx.Rngs(0), n_features= featurenum)
optimizer = nnx.Optimizer(policy, tx=optimizer,wrt=nnx.Param)


#for multiple episodes use a nested loop
for j in range(episodenum):
    inits, window, featurenum = envinit(10,10, seed, envnum)
    seed += 1
    for i in range(episodelength):
        inits, loss, collision = envstep(inits,policy,i,1,losses, goallog)
        collison = jnp.mean(collision)
        #envstates, statobjs, loss, collision, losses, randomgoals = envstep(inits,policy,i,1,losses,50)
        envstates = inits[0]
        statobjs = inits[1]
        print(j)
        if j == episodenum-1 and i > 1900:
            
            drawenv(envstates,statobjs,collision,window,env1frames)     
        loss, outputs = train_step(policy, envstates, optimizer, collision)

animate(env1frames)
drawplots(losses, episodelength)

print(goallog)

#example of multi-episode training run:
#for j in range(3):
#    policy, optimizer, envstates,statobjs = envinit(10,10,AgentNeuralNetwork,optimizer)
#    for i in range(300):
#        envstates, statobjs, loss, collision, window = envstep(i,j,envstates,statobjs,policy,losses, randomgoals,20)
#        train_step()
#    