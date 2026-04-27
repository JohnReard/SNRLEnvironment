
import jax
import optax
from flax import nnx as nnx
import jax.numpy as jnp 


devicecount = jax.device_count()
devices = jax.devices()

def addsclrs(x, y):
    return x+y

class Agent:
    #policy : agentneuralnetwork.AgentNeuralNetwork
    velocity : jnp.array
    angle : int
    #velocitylimit : int
    def __init__(self, policy):
        self.policy = policy
#uses sfm algorithm to pathfind for INDIVIDUAL humans
@jax.jit
def sfm(objstate, objstates, randomgoal):
    #check to see if goal needs to change
    #10% chance human changes goal
    
    packagedgoal = jnp.array([randomgoal[0],randomgoal[1],0])

    #random stopping
    packagedgoal
    #potentialgoal = objstate + packagedgoal
    goal = checkillegalstate(packagedgoal) #returns closest legal state
    #create movement towards goal
    objmove = goal - objstate  
    #jax.debug.print("objmove is: {z}", z=objmove)
    #check if movement brings you towards collidable object or static object
    distances = detectdistances(objstate,objstates)
    #if it does move away from object
    objtrans = objmove - jnp.sum(distances)
    #see whether agent stops to "talk" to other agent, or changes goal.


    clipinput = jnp.array([objmove[0],objmove[1]])
    mov = jnp.clip(clipinput, min=jnp.array([-2,-2]),max=jnp.array([2,2]))

    returnedstate = jnp.array([objstate[0] +  mov[0] ,objstate[1] + mov[1], objstate[2]])
    return returnedstate
@jax.jit
def checkillegalstate(state):#checks if point is in static object collision
    return state
@jax.jit
def detectdistances(objstate, otherobjs):
    dists = jax.vmap(lambda obj, othobj : othobj - obj, in_axes=(None,0))(objstate, otherobjs)
    return jnp.where(dists > 10, 0, dists)
@jax.jit
def detectcollision(movingstate, objectstate):
    leftarightb = (movingstate[0] - movingstate[2]) < (objectstate[0] + objectstate[2]) 
    leftbrighta = (objectstate[0] - objectstate[2]) < (movingstate[0] + movingstate[2]) 
    topabottomb = (movingstate[1]-movingstate[2])<(objectstate[1]+objectstate[2])
    topbbottoma = (objectstate[1] - objectstate[2])<(movingstate[1]+movingstate[2])
    xcollision = leftbrighta & leftarightb
    ycollision = topabottomb & topbbottoma
    return xcollision & ycollision
@jax.jit
def vmapcollisions(movingstate, objectstates):
    return jax.vmap(lambda agentstate, objectstates:jnp.where(detectcollision(agentstate,objectstates),objectstates, 0),in_axes=(None,0))(movingstate,objectstates)

@jax.jit
def statestep(envstate,currentaction,limits,randomgoal):#limits is a tuple
    #if I want to move multiple objects currentaction will be an array of actions and addedcoords will be vmapped
    #the collision detection will also be vmapped over all objects that have moved (maybe have moving objects as a separate argument?)
    #newcoords = addvelocity(envstate[1], currentaction) #envstate[1] is the agent velocity [500, 300] + [actionx, actiony]
    agentxy = jnp.array([envstate[1][0],envstate[1][1]])
    addedcoords = jax.vmap(lambda x, y : x + y)(agentxy, currentaction)# add agent transformation (action) to agent pos
    maxlim = jnp.array([limits[1]- envstate[1][2],limits[1]-envstate[1][2]])
    minlim = jnp.array([limits[0]+envstate[1][2],limits[0]+envstate[1][2]])
    newcoords = jnp.clip(addedcoords,min=minlim, max=maxlim) #make sure the new agent pos isn't outside of env limits assuming x = y for limits
    objectstates = envstate[2:len(envstate)]#for non-collidable goal
    #create an array of objectstates len for sfmcounter and map over it, otherwise all humans change direction at the same time
    newobjstates = jax.vmap(sfm,in_axes=(0,None,0))(objectstates,objectstates,randomgoal)
    #jax.debug.print("objmove is {x}", x=objmove[0])
    #sepobjstates = jax.vmap(lambda objstates: ([objstates[0],objstates[1]],objstates[2]))(objectstates)
    #addedobjstates = jax.vmap(lambda mov, prev : [prev[0]+mov[0],prev[1]+mov[1]])(objmove,sepobjstates[0])
    #newobjstates = jax.vmap(lambda adobj, sepobj: [adobj[0],adobj[1],sepobj[-]])
    
    #check for obj collisions

    agentstate = jnp.array([newcoords[0],newcoords[1],envstate[1][2]])
    #detect collisions
    collidedobjects = vmapcollisions(agentstate, objectstates)
    #correct collision states
    #cannot use if statements in vmapped func, so where used to separate 0s from non zeros
    #jnp.where(collidedobjects!=emptyarray, (agentstate[0]-objectstates[0], agentstate[1]-objectstates[1]),jnp.delete(collidedobjects,))
    #xmag, ymag = jax.vmap(lambda agn, obj: (agn[0]-round(obj[0],1), agn[1]-round(obj[1],1)),in_axes=(None,0))(agentstate,collidedobjects) #for one object
    
    #issue is here, not detecting enough objects/not mapping over right axes? collided objects is right shape, collision detection is working
    xobj = jax.vmap(lambda collidedobjects: collidedobjects[0],in_axes=(0))(collidedobjects)
    yobj = jax.vmap(lambda collidedobjects: collidedobjects[1],in_axes=(0))(collidedobjects)
    xobjs = jnp.sum(xobj)
    yobjs = jnp.sum(yobj)
    collision = ((xobjs>0)|(yobjs>0)).astype(int)
    newx = agentstate[0] + (-currentaction[0]* collision)
    newy = agentstate[1] + (-currentaction[1]* collision)
    
    
    #xdiff = jax.vmap(lambda objectstates ,agentstate, collision: ((agentstate[0] + agentstate[2] ) - (objectstates[0]+objectstates[2])) * collision,in_axes=(0,None,None))(collidedobjects, agentstate, collision)
    #ydiff = jax.vmap(lambda objectstates , agentstate, collision: ((agentstate[1] + agentstate[2] ) - (objectstates[1]+objectstates[2])) * collision, in_axes=(0,None,None))(collidedobjects, agentstate, collision)
    #xdiff = jnp.sum(xdiff)
    #ydiff = jnp.sum(ydiff)


    newcoords = jnp.array([newx,newy,agentstate[2]])
    array = jnp.array([envstate[0],newcoords])
    print("newobjstates: ", array.shape)
    newstate = jnp.concatenate([array, newobjstates], axis=0)
    print("shape of newstate is: ",newstate.shape)

    #print("prev state: ",envstate.at[1].get())
    #newstate = envstate.at[1].set(newcoords) #change index to be dynamic if other object is moving

    
    #detect collisions for newobjstates


    #collision=xobj.size
    #go through all the limits and clip?
    #might have to concatenate the agent state to goal state in another vmapped function 
    #if envstate[1][0] == envstate[0][0]:
    #    #env.goalreached()
    #    return newstate
    #else:
    collision = jnp.sum(collision)
    #jax.debug.print(jnp.shape(collision))
    return newstate, collision
@jax.jit
def addvelocity(a,b):#might not be vectorisable because might not be a pure function?
        addedcoords = jax.vmap(lambda x, y : x + y)(a, b)
        newcoords = jnp.clip(addedcoords,min=-600, max=600) #Clips x and y coords to -limits and limits #LIMITS ARE HARDCODED, CHANGE IN FUTURE
        return newcoords
@jax.jit
def lossfn(policy,states):
    limits = (0,600)
    goallocs, agentlocs = jax.vmap(extrgoalagentstate)(states)
    print("dist shape: ",goallocs.shape)
    print("shape in loss_fn: ", states.shape)
    outputs = act(policy,states)
    xdistances = jax.vmap(lambda x,y: x[0] - y[0]) (goallocs,agentlocs)
    ydistances = jax.vmap(lambda x,y: x[1] - y[1]) (goallocs,agentlocs)
    distances = jnp.reshape(jnp.array([xdistances,ydistances]),(len(xdistances),2))  
    distances = normalise(distances,limits)
        #reward = difference between what should happen and what did happen.
        # reward = optimal action - actual action
        #errors = jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances)
        #print(jnp.shape(collision))
        #lossx = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[0],collision))
        #lossy = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[1],collision))
        #loss = jnp.mean(lossx,lossy)
    loss = jnp.mean(jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances))
    return loss, outputs #n length vector of losses for each env
@jax.jit
def extrgoalagentstate(state):
    return state[0],state[1]
@jax.jit
def normalise(arg, limits):
    maxlim = limits[1]
    return jax.vmap(lambda arg : arg / maxlim)(arg)
@jax.jit
def act(policy,states):
    return jax.vmap(policy)(states)
@jax.jit
def train_step(policy, states, optimizer, collision):
    #states = jnp.array(jax.vmap(lambda states: [states[0:1]])(fullstates))
    print(states.shape)
    @jax.jit
    def lossfn(policy,states):
        limits = (0,600)
        goallocs, agentlocs = jax.vmap(extrgoalagentstate)(states)
        print("dist shape: ",goallocs.shape)
        print("shape in loss_fn: ", states.shape)
        outputs = act(policy,states)
        xdistances = jax.vmap(lambda x,y: x[0] - y[0]) (goallocs,agentlocs)
        ydistances = jax.vmap(lambda x,y: x[1] - y[1]) (goallocs,agentlocs)
        distances = jnp.reshape(jnp.array([xdistances,ydistances]),(len(xdistances),2))
        
        distances = normalise(distances,limits)
        #reward = difference between what should happen and what did happen.
        # reward = optimal action - actual action
        #errors = jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances)
        #print(jnp.shape(collision))
        #lossx = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[0],collision))
        #lossy = jnp.mean(jax.vmap(lambda x, y : x + y)(errors[1],collision))
        #loss = jnp.mean(lossx,lossy)
        loss = jnp.mean(jax.vmap(lambda x,y: optax.squared_error(x,y))(outputs,distances))
        return loss, outputs #n length vector of losses for each env

    valgrad = nnx.value_and_grad(lossfn,has_aux=True)
    
    (loss,actions),grads = valgrad(policy,states)
    optimizer.update(policy,grads)
    return loss, actions
    
    


#params = ann.init(key, initinput)
#creates an instance of the neural network (2 = no. of inputs, 100 = no. of neurons in hidden layer, 2 = no. of outputs, rng for random initial weights)

