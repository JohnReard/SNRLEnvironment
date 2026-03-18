import jax
import jax.numpy as jnp

a = jnp.array([0,1])
state1 = jnp.array([[30,30,6],
                    [20,20,5]])
state2 = jnp.array([[100,100,10],
                    [200,200,10]])
states = jnp.array([state1,state2])
actions = jnp.array([[50,5],[5,5]])
states = jnp.array([[8,22,2],[80,220,2],[9,24,1],[14,24,1]])
agent = jnp.array([7,10,4])


array= jnp.array([[0,0,0],[1,4,1],[0,0,0]])
#                                                        in_axes=(0,None))(array,agentstate)
array = jnp.trim_zeros(array)
print(array)
#array = jnp.ravel(array)
#result = jax.vmap(lambda array: jnp.where(array[0]>2,array[0],0))(array)
#result = jax.vmap(lambda array, agentstate: (((array[0]+array[2])-(agentstate[0]+agentstate[2]))+agentstate[1], ((array[1]+array[2])-(agentstate[1]+agentstate[2]))+agentstate[1]),in_axes=(0,None))(result,agentstate)

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