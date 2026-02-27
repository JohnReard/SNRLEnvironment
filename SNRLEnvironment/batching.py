import jax
import jax.numpy as jnp

seed = 1000
key = jax.random.key(seed)
seed = 2000

@jax.jit #this calls envnum times
def create_env(randint):
    state = jnp.clip(randint*100,min=0, max=600)
    returnedstate = jnp.array(state)
    return returnedstate
@jax.jit
def return_states(randints):
    envbatch = jax.vmap(create_env)(randints)
    return envbatch
#this calls once, can't be jitted because arrays need static shape
def create_envbatch(key, envnum):
    #find a way to do this for an indeterminate number of objects?
    goalrandints = jax.random.normal(key, shape=(envnum,2))
    subkey, key = jax.random.split(key)
    agentrandints = jax.random.normal(key, shape=(envnum,2))
    goalstates = return_states(goalrandints)# [n*[x,y] ]
    agentstates = return_states(agentrandints)# [n*[x,y] ]
    #returns initialstates, idealstates (idealstates is the agent on the goal, used for loss func)
    return jnp.reshape(jnp.array([goalstates,agentstates]),(envnum,2,2)), jnp.reshape(jnp.array([goalstates,goalstates]),(envnum,2,2))
