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
def create_centres(objsize, key, limits,statnum):
    centres = jax.random.uniform(key,shape=(statnum,2),minval=limits[0]+objsize,maxval=limits[1]+objsize)
    return centres
    
@jax.jit
def return_vertices(centre,objsize,rotation):
    jax.debug.print("{x} is objsize",x=objsize)
    theta = rotation * (jnp.pi/180)
    initvert = ((centre[0] - objsize[0], centre[1] - objsize[1]),
                (centre[0] + objsize[0], centre[1] - objsize[1]),
                (centre[0] - objsize[0], centre[1] + objsize[1]),
                (centre[0] + objsize[0], centre[1] + objsize[1]))
    #rotate shape, theta is angle in RADIANS
    newvertices = []
    dists = []
    for vertex in initvert:
        newx = (((centre[0] - vertex[0]) * jnp.cos(theta)) - ((centre[1] - vertex[1]) * jnp.sin(theta)))+ centre[0]
        newy = (((centre[1] - vertex[1]) * jnp.sin(theta)) + ((centre[0] - vertex[0]) * jnp.cos(theta)) + centre[1])
        a = centre[0]-vertex[0]
        b = centre[1]-vertex[1]
        dists.append([((centre[0] - vertex[0]) * jnp.cos(theta)),((centre[1] - vertex[1]) * jnp.sin(theta))])
        #distance should be signed?? if it is it can transform vertex to rotate
        newvertices.append([newx,newy])
    vertices=jnp.array(newvertices)
    print("my shape ", jnp.array(dists).shape)
    #the y is the same for top left and top right, top left should decrease and top right should increase?
    jax.debug.print("{x} is init, {y} is rotated", x= initvert, y=newvertices)
    return vertices, jnp.array(dists)

@jax.jit
def create_vertices(rotations, centres, objsizes):
    #vertices are statobjnum in shape
    print(centres.shape,objsizes.shape,rotations.shape)
    vertices=[]
    vertices, dists = jax.vmap(return_vertices)(centres,objsizes,rotations)#statobjnum shape
    return vertices, dists
@jax.jit
def return_statobjs(vertices, centres):
    return jax.vmap(lambda c, v :jnp.array([v[0],v[1],v[2],v[3],c]))(centres,vertices)
@jax.jit
def return_states(randints,objrad, objsizes, rotations, limits, centres):
    #rotation will act as a multiplier on the value of the vertices, they will increment in the y by the inverse of the x
    #centres should be envnum, objnum
    vertices, dists = jax.vmap(create_vertices)(rotations,centres,objsizes)
    envbatch = jnp.array(jax.vmap(create_env,in_axes=(0,None))(randints,objrad))
    print("envrand: ",envbatch.shape)
    print(centres[0])
    #need to map over centres as well

    statobjs = jax.vmap(return_statobjs,in_axes=(0,0))(vertices, centres)#shape is (vertex1, vertex2, vertex3) if more vertices exist ALL objects must have same number of elements, but can be 0
    
    #statobjs = centres, vertices
    return envbatch, statobjs, dists
#this calls once, can't be jitted because arrays need static shape
@jax.jit
def detectillegal(obja, objb):
    #is obja collided with objb?
    leftarightb = (obja[0] - obja[2]) < (objb[0] + objb[2]) 
    leftbrighta = (objb[0] - objb[2]) < (obja[0] + obja[2]) 
    topabottomb = (obja[1] - obja[2]) < (objb[1] + objb[2])
    topbbottoma = (objb[1] - objb[2]) < (obja[1] + obja[2])
    xcollision = leftbrighta & leftarightb
    ycollision = topabottomb & topbbottoma
    return xcollision & ycollision
@jax.jit
def vmapcollisions(obja, objectstates):
    #for every state in an environment
    #objectstates contains obja
    #map over all states and see if they collide with obja, return collided state
    return [jnp.sum(jax.vmap(lambda obj, objectstates:jnp.where(detectillegal(obj,objectstates) & (obj != objectstates),
                                                       1, 0),
                                                       in_axes=(None,0))(obja,objectstates)),0]
@jax.jit
def replaceillegalstates(newstate,oldstate,arg):
    #map over another axis again
    return jnp.where()
@jax.jit  
def detectcollisions(environment):
    #for every environment, for every state, check whether it collides with all other states
    return jax.vmap(lambda obj, allobjs: vmapcollisions(obj,allobjs),in_axes=(0,None))(environment,environment)
def correctillegalstates(seed, environments, objnum, objsize, limits,envnum):
    seed += 1
    print("seed:",seed)
    key = jax.random.key(seed)
    randints = jax.random.uniform(key, shape=(envnum,objnum+2,2),minval=limits[0]+objsize,maxval=limits[1]-objsize)
    collisions = jax.vmap(detectcollisions)(environments)
    #onehot encode legal and illegal states
    #booleancollisions = jax.vmap(lambda col : jnp.where(col > 0, 1, 0))(collisions)
    print("collisions: ",collisions[0])
    print("randints: ",randints[0])
    randints=jnp.array(randints)
    #collisions=jnp.reshape(jnp.array(collisions),(randints.shape))
    #needs to be vmapped conditional func not jnp where, otherwise replacement coords arent mapped across
    print(environments[0])
    randints = jnp.array(jax.vmap(create_env,in_axes=(0,None))(randints,objsize))
    newstates = jax.vmap(lambda oldstate, newstate, col: jnp.where(col != 0 , newstate, oldstate))(environments, randints, collisions)
    
    #newstates = jax.vmap(replaceillegalstates)(randints,environments, booleancollisions)
    #newstates = jax.vmap(lambda obj, randcoords, env:jnp.where(obj[0] != 0 , randcoords, obj))(collisions, randints, environments)
    print("newstates: ", newstates[0])
    return newstates #returns environments without 0s
def countillegalenvironments(markedstates):
    return jnp.where(markedstates == 0, 1, 0)
	
def recursivecorrection(environments, objnum, objsize, limits, seed, envnum):
	#returns environments with illegal states replaced with new random states
    print("envs: ",environments)
    newstates = correctillegalstates(seed, environments, objnum,objsize,limits,envnum)
    #detects collided states and returns them as 0s.
    markedenvs = jax.vmap(detectcollisions)(newstates)
    print("marked envs: ", markedenvs)
    #counts no. of 0s (i.e how many illegal states there are)
    numillegal = jnp.sum(jnp.where(countillegalenvironments(markedenvs), 1, 0))
    print("numillegal: ",numillegal)
    #numillegal = jnp.sum(numillegal)
    if numillegal == 0:
        return newstates
    else:
	    recursivecorrection(newstates, objnum, objsize, limits,seed+1,envnum)


def create_envbatch(seed, envnum, limits, objnum, objrad,staticnum):
    key = jax.random.key(seed)
    maxstatsize = 70
    #create envnum partitions, these are arrays containing
    #find a way to do tehis for an indeterminate number of objects?
    initstateints = jax.random.uniform(key, shape=(envnum,objnum+2,2),minval=limits[0]+objrad,maxval=limits[1]-objrad)
    #generate random sizes, positions for static objects and a rotation for these shapes
    centres = jax.random.uniform(key,shape=(envnum,staticnum,2),minval=limits[0]+maxstatsize,maxval=limits[1]+maxstatsize)
    rotations = jax.random.uniform(key,shape=(envnum,staticnum),minval=0,maxval=360)
    objsizes = jax.random.uniform(key, shape=(envnum,staticnum,2),minval=5,maxval=40)
    
    envstates, statobjs, dists = return_states(initstateints,objrad,objsizes,rotations,limits,centres)# [n*[objnum+2* [x,y] ] [*4[x,y],(centrepoint)[x,y]]
    
    print("init agent state: ",envstates[0][1])
    #rejection sampling
    #check if any object is outside of limits or in illegal state
    initstateints = jax.random.uniform(key, shape=(envnum,objnum+2,2),minval=limits[0]+objrad,maxval=limits[1]-objrad)
    returnedstates = recursivecorrection(envstates, objnum, objrad, limits, seed+1,envnum)
    return returnedstates, statobjs, rotations, dists
    #return jnp.reshape(envstates,(envnum,objnum+2,3)), statobjs,rotations, dists
key = jax.random.key(1204)