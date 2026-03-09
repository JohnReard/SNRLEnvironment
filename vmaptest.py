import jax
import jax.numpy as jnp

a = jnp.array([0,1])
state1 = jnp.array([[30,30,6],
                    [20,20,5]])
state2 = jnp.array([[100,100,10],
                    [200,200,10]])
states = jnp.array([state1,state2])
actions = jnp.array([[5,5],[5,5]])
states = jnp.array([[8,22,2],[80,220,2]])
agent = jnp.array([7,24,5])


agentstate = jnp.array(agent)
@jax.jit
def f(state,agentstate):
     #state will not have agent (state[1]) in it or goal (state[0])
     #agentstate is the state of the agent AFTER movement, we clip it here.
    objx = jnp.array(state[0])
    objy = jnp.array(state[1])
    objrad = jnp.array(state[2])
    legalarray = jnp.where(((objx +  objrad < agentstate[0] - agentstate[2]) & (objx - objrad < agentstate[0] - agentstate[2]))
            |((objy -  objrad > agentstate[1] + agentstate[2]) & (objy +  objrad > agentstate[0] - agentstate[2])),jnp.array([objx,objy,objrad]),0)
    jnp.clip(agentstate, min=)
    return legalarray
    


@jax.jit
def fjit(states,agentstate):
    return jax.vmap(f,in_axes=(0,None))(states,agentstate)
     
result = fjit(states,agentstate)
result = jax.vmap(lambda arr: arr[2] > 0)(result)
print(result)
#print(jnp.nonzero(result))
#myfunc = jax.vmap(lambda newarray: jnp.where(newarray[2]==0))
#print(myfunc(result))
#result = jax.vmap(lambda x, y: x + y, in_axes=(2,0))(state, actions)
#when the in axes is specified you are specifying that that number axis or dimension is being iterated through, i.e:
#print(result)
#is: | 30 , 40 |
#    | 10 , 20 |   --> [ [30, 10] , [41, 21] ]
#     ---------
#     + 0 , +1
#whereas if in_axes were (0,0):
#    | 30 , 40 | + 0
#    | 10 , 20 | + 1   --> [ [30, 40] , [11, 21] ]
#
#so essentially we are mapping over the first dim (the row) with 0 and the second dim (the column) with 1

#x = jnp.array([1.0, 2.0, 3.0])  # batched
#y = jnp.array([10.0])            # not batched (broadcast to all)

#print(jax.vmap(lambda x, y : x + y, in_axes=(0, None) )(x, y))  # → [11., 12., 13.]