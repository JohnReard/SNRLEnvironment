import jax
import jax.numpy as jnp
import jax.lax as lax

#seed = 1000
#key = jax.random.key(seed)
#seed = 2000
@jax.jit
def createarray(x):
    return jnp.array([])

@jax.jit #this calls envnum times
def create_env(randint, objrad):
    states = jax.vmap(lambda state,objrad:jnp.array([state[0],state[1],objrad]),in_axes=(0,None))(randint,objrad) #randints shape: [objnum+2 * [x, y]]
    #unfortunately can't parallelise adding elements to array (or even pytrees)
    return states #returned states is objnum + 2 * [x,y]
@jax.jit
def assignpartitions(state):
    #32 partitions
    partitionnum = lax.round(((state[0]+state[1])/37.5))
    #returnedpart = partitions[partitionnum.astype(int)]
    return partitionnum
@jax.jit
def return_states(randints,objrad):
    envbatch = jax.vmap(create_env,in_axes=(0,None))(randints,objrad)
    return envbatch
#this calls once, can't be jitted because arrays need static shape
def create_envbatch(key, envnum, limits, objnum, objrad ):
    #create envnum partitions, these are arrays containing
    #find a way to do this for an indeterminate number of objects?
    initstateints = jax.random.uniform(key, shape=(envnum,objnum+2,2),minval=limits[0]+objrad,maxval=limits[1]-objrad)
    envstates = return_states(initstateints,objrad)# [n*[objnum+2* [x,y] ]
    print("init agent state: ",envstates[0][1])
    return jnp.reshape(envstates,(envnum,objnum+2,3))
key = jax.random.key(1204)