import os
import psutil
coreno = os.cpu_count()
#JAX_NUM_CPU_DEVICES=2
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16'

import random
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
#from jax.experminental.pallas import 
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P

import matplotlib.pyplot as plt

devicecount = jax.local_device_count()
devices = jax.devices()

seed = 1001
key = jax.random.key(seed)

i=0
print("local device count is: ",devicecount)
arr1 = jnp.array([1, 2])
arr2 = jnp.array([3, 4])
a = jnp.array([5, 5])
b = jnp.array([10,10])
ab = jnp.array([a,b])
#print(myfunc(a,b))
addedcoords = jax.vmap(lambda x, y : x + y)(a,b)
def myfunc(c,d):#vmap has to map over each argument's FIRST dimension individually
    first = jnp.array([c[0] + d[0]])
    second = jnp.array([c[1]+ d[1]])
    output = jnp.array([first,second])
    return output
#myarray1 = jnp.array([arr1,a])
#myarray2 = jnp.array([arr2,b])
#result = jax.vmap(myfunc)([arr1,a],[arr2,b])

#print("result: ",result)
#print("ndim of result: ",result.ndim)
#print("ndim of input:", myarray1.ndim)
#addedcoords = jnp.clip(addedcoords,min=0, max=2)
#[coordinates +1] [coordinates+1] 

#a = jnp.arange(1).reshape(jax.device_count())
print("a: ", a)
#these mesh and inp specs are equal to 
mesh = jax.make_mesh((jax.device_count(),), 'cores') #first argument is the dimensions of the mesh, we have x devices and 1 blocks on each
#jax.create_ten
jax.set_mesh(mesh)
@jax.shard_map(in_specs=P('cores'), out_specs=P('cores'))
def shardedfunc(argument):
    values = psutil.cpu_percent(percpu=True) 
    print(values)
    return argument + 1
data = jax.device_put(jnp.arange(jax.device_count()),P('cores'))
print(shardedfunc(data))
while i < 10000:
    shardedfunc(data)
    i+=1
print("number of cores: ", coreno)

