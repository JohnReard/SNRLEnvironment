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
episodelength = 300


policy, optimizer, envstates,statobjs, window = envinit(10,10,AgentNeuralNetwork,optimizer, 8020)


#for multiple episodes use a nested loop
for i in range(episodelength):
    envstates, statobjs, loss, collision, losses = envstep(i,envstates,statobjs,policy,losses)
    drawenv(envstates,statobjs,collision,window,env1frames)
    loss, outputs = train_step(policy, envstates, optimizer, collision)

animate(env1frames)
drawplots(losses, episodelength)


#example of multi-episode training run:
#for j in range(3):
#    policy, optimizer, envstates,statobjs = envinit(10,10,AgentNeuralNetwork,optimizer)
#    for i in range(300):
#        envstates, statobjs, loss, collision, window = envstep(i,envstates,statobjs,policy)
#        train_step()
#    