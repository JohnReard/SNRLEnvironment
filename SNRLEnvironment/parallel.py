import jax
from jax import pmap
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from functools import partial

devicecount = jax.device_count()
devices = jax.devices()
print("device count:",devicecount)

array1 = jnp.ones((1,100))
array2 = jnp.ones((1,100))

#mesh = Mesh(devices.reshape(1, 1), ('x', 'y'))
@partial(shard_map,mesh=mesh,in_specs=(('x', 'y'),('x', 'y')),out_specs=('x', 'y'))
def shardadd(x,y):
    return x + y

output = pmap(lambda x, y: x + y)(array1, array2)
output2 = jax.lax.psum(output,devicecount)
print("output:",output, "sum:", output2, "shardadd:",shardadd(array1,array2))