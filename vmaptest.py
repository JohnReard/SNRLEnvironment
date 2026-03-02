import jax
import jax.numpy as jnp

a = jnp.array([0,1])
state1 = jnp.array([[30,40],
                    [10,20]])
state2 = jnp.array([[100,100],
                    [200,200]])
states = jnp.array([state1,state2])
actions = jnp.array([[5,5],[5,5]])

result = jax.vmap(lambda x, y: x + y, in_axes=(2,0))(states, actions)
#when the in axes is specified you are specifying that that number axis or dimension is being iterated through, i.e:
print(result)
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