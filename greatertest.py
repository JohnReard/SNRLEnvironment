import jax
import jax.numpy as jnp

a = jnp.array([[200,100],[300,300]])
b = jnp.array([200,100])
print(jnp.greater(a,b))
