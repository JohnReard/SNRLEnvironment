import jax
import jax.numpy as jnp
import jax.lax as lax
import math
#seed = 1000
#key = jax.random.key(seed)
#seed = 2000
@jax.jit #this calls envnum times
def create_env(randint, objrad):
    states = jax.vmap(lambda state,objrad:jnp.array([state[0],state[1],objrad]),in_axes=(0,None))(randint,objrad) #randintsmob shape: [objnum+2 * [x, y]]
    #unfortunately can't parallelise adding elements to array (or even pytrees)
    return states #returned states is objnum + 2 * [x,y]
@jax.jit
def create_centres(objsize, key, limits,statnum):
    centres = jax.random.uniform(key,shape=(statnum,2),minval=limits[0]+objsize,maxval=limits[1]+objsize)
    return centres
    
@jax.jit
def return_vertices(centre,objsize,rotation):
    print("shape of centre: ",centre.shape)
    theta = rotation * (jnp.pi/180)
    
    initvert = ((centre[0] - objsize[0], centre[1] - objsize[1]),#top left
                (centre[0] + objsize[0], centre[1] - objsize[1]),#top right
                (centre[0] - objsize[0], centre[1] + objsize[1]),#bottom left
                (centre[0] + objsize[0], centre[1] + objsize[1]),#bottom right
                (rotation, rotation)) #rotation stored for collision detection
    #rotate shape, theta is angle in RADIANS
    newvertices = []
    for vertex in initvert: # may not be reversed! make sure you're ready to undo
        newx = (((vertex[0] - centre[0] ) * jnp.cos(theta)) - ((vertex[1] - centre[1] ) * jnp.sin(theta)))+ centre[0]
        newy = ((vertex[1] - centre[1]) * jnp.cos(theta)) + ((vertex[0] - centre[0]) * jnp.sin(theta)) + centre[1]
        #distance should be signed?? if it is it can transform vertex to rotate
        newvertices.append([newx,newy])
    vertices=jnp.array(newvertices)
    #the y is the same for top left and top right, top left should decrease and top right should increase?
    #jax.debug.print("{x} is init, {y} is rotated", x= initvert, y=newvertices)
    return vertices
@jax.jit
def return_rects_per_env(centres,objsizes,rotations):
    #centres, objsizes and rotations will be statnum in length
    return jax.vmap(return_vertices)(centres, objsizes,rotations)

@jax.jit
def create_vertices(rotations, centres, objsizes):
    #vertices are statobjnum in shape
    print(centres.shape,objsizes.shape,rotations.shape)
    vertices=[]
    jax.debug.print("centres: {x} ", x=centres[0])
    vertices = jax.vmap(return_vertices)(centres,objsizes,rotations)#statobjnum shape
    return vertices
@jax.jit
def return_statobjs(vertices, centres):
    vert = jnp.array(vertices)
    jax.debug.print("vert = {x}",x=vertices[0])
    jax.debug.print("vert = {x}",x=vertices[0])
    print("beep: ",centres.shape,vert.shape)
    jax.debug.print("centres: {x} vertices: {y}",x=centres,y=vertices)
    #map over v again
    
    return jax.vmap()(centres,vert)
#@jax.jit
#def mapoverrects(centres, vertices):
#    return jax.vmap(lambda c, v : jnp.array([v[0],v[1],v[2],v[3],c]),)(centres,vert)
@jax.jit
def return_states(randints,objrad, objsizes, rotations, limits, centres):
    #rotation will act as a multiplier on the value of the vertices, they will increment in the y by the inverse of the x
    #centres should be envnum, objnum
    vertices = jax.vmap(create_vertices)(rotations,centres,objsizes)
    envbatch = jnp.array(jax.vmap(create_env,in_axes=(0,None))(randints,objrad))
    print("envrand: ",envbatch.shape)
    print(centres[0])
    #need to map over centres as well
    print("vertices: ",vertices)
    
    #print(statobjs.shape)
    #statobjs = jax.vmap(return_statobjs,in_axes=(0,0))(vertices, centres)#shape is (vertex1, vertex2, vertex3) if more vertices exist ALL objects must have same number of elements, but can be 0
    #statobjs = centres, vertices
    return envbatch, vertices
#this calls once, can't be jitted because arrays need static shape
@jax.jit
def detectillegalmob(obja, objb):
    #is obja collided with objb?
    leftarightb = (obja[0] - obja[2]) < (objb[0] + objb[2]) 
    leftbrighta = (objb[0] - objb[2]) < (obja[0] + obja[2]) 
    topabottomb = (obja[1] - obja[2]) < (objb[1] + objb[2])
    topbbottoma = (objb[1] - objb[2]) < (obja[1] + obja[2])
    xcollision = leftbrighta & leftarightb
    ycollision = topabottomb & topbbottoma
    return xcollision & ycollision
@jax.jit
def detectillegalstat(obja, objb_unrotated): #when calling for mobile vs stat collision define the edges of the circle before they are passed into the function
    #rotationdif is how objb is rotated from obja's rotation as a starting point
    rotation = obja[4][0]#obja is static, objb is mobile
    leftb = [objb_unrotated[0] - objb_unrotated[2], objb_unrotated[1] ] #is a scalar
    rightb = [objb_unrotated[0] + objb_unrotated[2], objb_unrotated[1] ]
    topb = [objb_unrotated[0] , objb_unrotated[1] - objb_unrotated[2]]
    bottomb = [objb_unrotated[0] ,objb_unrotated[1] + objb_unrotated[2]]
    acentre = (((obja[0] - obja[1])/2)+obja[0],((obja[0] - obja[2])/2)+obja[0])#might end up with floating point errors, maybe store centre in object state?
    angle = rotation * (jnp.pi/180)
    #rotate the mobile object
    #pass in objb
    edges = jnp.array([leftb,rightb,topb,bottomb])
    centreb = objb_unrotated[:1]
    print(jnp.array(leftb).shape)
    rotobjbxs = jax.vmap(lambda edge, centreb, theta : (((edge[0] - centreb[0] ) * jnp.cos(theta)) - ((edge[1] - centreb[1] ) * jnp.sin(theta)))+ centreb[0] ,in_axes=(0,None, None))(edges,centreb, angle)
    rotobjbys = jax.vmap(lambda edge, centreb, theta : (((edge[1] - centreb[1] ) * jnp.cos(theta)) + ((edge[0] - centreb[0] ) * jnp.sin(theta)))+ centreb[1] ,in_axes=(0,None, None))(edges,centreb, angle)

    #objbxs = jax.vmap(lambda centre, vertex, theta:
    #                ((((centre[0] - vertex[0]) * jnp.cos(theta)) - ((centre[1] - vertex[1]) * jnp.sin(theta)))+ centre[0]),in_axes=(None,0,None))(acentre,objb_unrotated, angle)
    #                 
    #objbys = jax.vmap(lambda centre, vertex, theta:
    #                    (((centre[1] - vertex[1]) * jnp.sin(theta)) + ((centre[0] - vertex[0]) * jnp.cos(theta))) + centre[1],
    #                    in_axes=(None,0,None))(bcentre,objb_unrotated, angle)
    #objb = jnp.stack([objbxs,objbys],axis=1)
    #jax.debug.print("objb shape: {x} ", x=objb)
    #distance should be signed?? if it is it can transform vertex to rotate

    leftarightb = (obja[0][0]) < (rotobjbxs[1]) #top left a < top right b
    leftbrighta = (rotobjbxs[0]) < (obja[1][0]) # top left b < top right a
    topabottomb = (obja[0][1]) < (rotobjbys[3]) # bottom left a < bottom left right
    topbbottoma = (rotobjbys[2]) < (obja[2][1])
    xcollision = leftbrighta & leftarightb
    ycollision = topabottomb & topbbottoma
    return xcollision & ycollision
@jax.jit
def vmapcollisions(obja, objectstates):
    #for every state in an environment
    #objectstates contains objaprint
    #map over all states and see if they collide with obja, return collided state
    
    return [jnp.sum(jax.vmap(lambda obj, objectstates:jnp.where(detectillegalmob(obj,objectstates) & (obj != objectstates),
                                                       1, 0),
                                                       in_axes=(None,0))(obja,objectstates)),0]
@jax.jit
def vmapcollisionsstat(statobj, mobobjs):
    #for every state in an environment
    #objectstates contains obja
    
    #map over all states and see if they collide with obja, return collided state
    #a is statobj, b is mob

    return [jnp.sum(jax.vmap(lambda obj, objstates:jnp.where(detectillegalstat(obj,objstates) & (obj[0] != objstates[0]),
                                                       1, 0),
                                                       in_axes=(None,0))(statobj,mobobjs)),0]
@jax.jit  
def detectcollisions(environment):
    #for every environment, for every state, check whether it collides with all other states
    return jax.vmap(lambda obj, allobjs: vmapcollisions(obj,allobjs),in_axes=(0,None))(environment,environment)
@jax.jit
def detectstatcollisions(statobjs, mobobj):
    return jax.vmap(lambda obj, allobjs: vmapcollisionsstat(obj,allobjs),in_axes=(0,None))(statobjs,mobobj)
@jax.jit
def return_vert_centres(vertices, centres):
    #called per environments
    return jax.vmap(return_statobjs)(vertices,centres)
def return_rectangles(key,envnum,statnum,limits,maxstatsize):
    centres = jax.random.uniform(key,shape=(envnum,statnum,2),minval=limits[0]+maxstatsize,maxval=limits[1]+maxstatsize)
    rotations = jax.random.uniform(key,shape=(envnum,statnum),minval=0,maxval=360)
    objsizes = jax.random.uniform(key, shape=(envnum,statnum,2),minval=5,maxval=40)
    vertices = jax.vmap(return_rects_per_env)(centres,objsizes,rotations)#returns envnum, statnum rects
    #map over envs
    print("final shape is: ", jnp.array(vertices).shape)
    
    #statobjs = jax.vmap(return_vert_centres,in_axes=(0,0))(vertices, centres)#packages verts with their centres
    return vertices


def correctillegalstates(seed, environments, objnum, objsize, limits,envnum, statnum,maxstatsize,oldvertices):
    seed += 1
    print("seed:",seed)
    key = jax.random.key(seed)
    
    #create random integers for moving objects
    randintsmob = jax.random.uniform(key, shape=(envnum,objnum+2,2),minval=limits[0]+objsize,maxval=limits[1]-objsize)
    randintsmob=jnp.array(randintsmob)
    
    
    #create array of objects where 0 = no collision 1 = collision
    collisions = jax.vmap(detectcollisions)(environments)
    collisionsstat = jnp.array(jax.vmap(detectstatcollisions)(oldvertices,environments))
    
    #random states for mobile objects packaged with objsize
    randintsmob = jnp.array(jax.vmap(create_env,in_axes=(0,None))(randintsmob,objsize))
    print("Mobile shapes: ", jnp.array(environments).shape, jnp.array(randintsmob).shape, jnp.array(collisions).shape)
    newstates = jax.vmap(lambda oldstate, newstate, col: jnp.where(col != 0 , newstate, oldstate))(environments, randintsmob, collisions)
    
    randvertices = return_rectangles(key,envnum,statnum,limits,maxstatsize)
    randvertices = jnp.array(randvertices)
    print("randvertices: ", randvertices.shape)

    collisionsstat = jnp.reshape(jnp.array(collisionsstat), (envnum,statnum,2))
    print("shapes of args: ",randvertices.shape," " ,oldvertices.shape," ", collisionsstat.shape )
    collisionsstat = jnp.expand_dims(collisionsstat, axis=2)
    print("collisionsstat: ",collisionsstat)
    newvertices = jnp.array(jax.vmap(lambda oldstate, newstate, col: jnp.where(col != 0 , newstate, oldstate))(randvertices, oldvertices, collisionsstat))
    #newstates = jax.vmap(replaceillegalstates)(randintsmob,environments, booleancollisions)
    #newstates = jax.vmap(lambda obj, randcoords, env:jnp.where(obj[0] != 0 , randcoords, obj))(collisions, randintsmob, environments)
    print("newstates: ", newstates[0])
    return newstates, newvertices #returns environments without 0s
def countillegalenvironments(markedstates):
    return jnp.where(markedstates == 0, 1, 0)
	
def recursivecorrection(environments, objnum, objsize, limits, seed, envnum,statnum,maxstatsize,vertices):
	#returns environments with illegal states replaced with new random states
    #print("envs: ",environments)
    newstates, newvertices = correctillegalstates(seed, environments, objnum,objsize,limits,envnum,statnum, maxstatsize,vertices)
    #detects collided states and returns them as 0s.
    markedenvs = jax.vmap(detectcollisions)(newstates)
    markedstats = jax.vmap(detectstatcollisions)(newvertices,newstates)
    print("marked envs: ", markedenvs)
    #counts no. of 0s (i.e how many illegal states there are)
    numillegalmob = jnp.sum(jnp.where(countillegalenvironments(markedenvs), 1, 0))
    numillegalstat = jnp.sum(jnp.where(countillegalenvironments(markedstats), 1, 0))
    print("numillegal: ",numillegalmob)
    #numillegal = jnp.sum(numillegal)
    if numillegalmob == 0 and numillegalstat == 0:
        return newstates, newvertices
    else:
	    recursivecorrection(newstates, objnum, objsize, limits,seed+1,envnum,statnum, maxstatsize,vertices)


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
    
    envstates, statobjs = return_states(initstateints,objrad,objsizes,rotations,limits,centres)# [n*[objnum+2* [x,y] ] [*4[x,y],(centrepoint)[x,y]]
    print("statobjs is: ",statobjs[0])
    print("init agent state: ",envstates[0][1])
    #rejection sampling
    #check if any object is outside of limits or in illegal state
    #initstateints = jax.random.uniform(key, shape=(envnum,objnum+2,2),minval=limits[0]+objrad,maxval=limits[1]-objrad)
    returnedmobstates, returnedstatstates = recursivecorrection(envstates, objnum, objrad, limits, seed+1,envnum,staticnum, maxstatsize,statobjs)
    
    return returnedmobstates, returnedstatstates
    #return jnp.reshape(envstates,(envnum,objnum+2,3)), statobjs,rotations, dists
key = jax.random.key(1204)