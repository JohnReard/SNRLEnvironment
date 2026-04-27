import training
from training import envstep,envinit,drawenv,animate,drawplots
from Env1 import train_step
from flax import nnx as nnx
import optax
#import your own agent implementation
from agentneuralnetwork import AgentNeuralNetwork

optimizer = optax.adam(learning_rate=0.08)
env1frames = []
losses = []
goallog = []
episodelength = 300
episodenum = 3

envnum = 600
seed = 81
sfmcounter = 100


inits, optimizer, window, policy = envinit(10,10,AgentNeuralNetwork,optimizer, seed, envnum)


#for multiple episodes use a nested loop
for j in range(episodenum):
    for i in range(episodelength):
        inits, loss, collision = envstep(inits,policy,i,1,losses, goallog)
        #envstates, statobjs, loss, collision, losses, randomgoals = envstep(inits,policy,i,1,losses,50)
        envstates = inits[0]
        statobjs = inits[1]
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