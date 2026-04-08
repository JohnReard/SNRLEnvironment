import jax
import jax.numpy as jnp
import jax.lax as lax
import math
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
def create_statobj(objsizes, rotations, key):
    jax.random.uniform(key,minval=limits[0]+,maxval=40)
    objsizes

@jax.jit
def return_states(randints,objrad, objsizes, rotations,centres, limits):
    #rotation will act as a multiplier on the value of the vertices, they will increment in the y by the inverse of the x
    centres = jax.vmap(create_statobj, in_axes=(0,0,None))(objsizes,rotations,key)
    vertices = jax.vmap(create_vertices)(rotations,centres,objsizes)
    envbatch = jax.vmap(create_env,in_axes=(0,None))(randints,objrad)
    size
    statobjs = #shape is (vertex1, vertex2, vertex3) if more vertices exist ALL objects must have same number of elements, but can be 0
    return envbatch, statobjs
#this calls once, can't be jitted because arrays need static shape
def create_envbatch(key, envnum, limits, objnum, objrad,staticnum):
    #create envnum partitions, these are arrays containing
    #find a way to do tehis for an indeterminate number of objects?
    initstateints = jax.random.uniform(key, shape=(envnum,objnum+2,2),minval=limits[0]+objrad,maxval=limits[1]-objrad)
    #generate random sizes, positions for static objects and a rotation for these shapes
    rotations = jax.random.uniform(key,shape=(envnum,staticnum),minval=0,maxval=2*math.pi)
    objsizes = jax.random.uniform(key, shape=(envnum,staticnum),minval=5,maxval=40)
    envstates, statobjs = return_states(initstateints,objrad,objsizes,rotations)# [n*[objnum+2* [x,y] ] [*4[x,y],(centrepoint)[x,y]]
    print("init agent state: ",envstates[0][1])
    return jnp.reshape(envstates,(envnum,objnum+2,3))
key = jax.random.key(1204)