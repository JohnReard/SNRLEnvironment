import jax
import jax.numpy as jnp

seed = 1002
key = jax.random.key(seed)


def create_env(envinits, key):
    key, subkey = jax.random.split(key)
    subkey1, subkey2 = jax.random.split(subkey)
    subkey1 = jnp.clip(int(subkey1), min= 0, max= 600)
    subkey2 = jnp.clip(int(subkey2), min= 0, max= 600)
    return jnp.array([jnp.array(jnp.array([subkey1]),jnp.array([subkey2]))])
def create_envbatch(key, envnum):
    envinits = jnp.zeros(envnum)
    envbatch = jax.vmap(create_env, out_axes=envnum, in_axes=(None,None))(envinits,key)
    print(envbatch)
create_envbatch(key, 5)