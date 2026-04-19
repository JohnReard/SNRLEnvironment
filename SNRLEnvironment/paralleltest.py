import os
import psutil
coreno = os.cpu_count()
#JAX_NUM_CPU_DEVICES=2
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16'

import random
import jax
from jax import lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
#from jax.experminental.pallas import 
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P

import matplotlib.pyplot as plt


myarray = jnp.array([[0,2], [2,10], [10,4], [2,0]])
array2 = jnp.array([[1,3],[0,1],[0,1],[2,5]])
myrange = len(myarray) - len(array2)
arrayarg = jnp.append(array2,jnp.arange(myrange))
def gennew(val):
    seed = val[0]
    print(val)
    key = jax.random.key(seed)
    key, disc = jax.random.split(key)
    newval = jax.random.uniform(key, shape=(1,2), maxval=10, minval=0)
    newval = newval[0].astype(int)
    return lax.cond(newval[0] > 2, lambda x: x, gennew(newval), newval)
    
def myfunc(element):
    return jnp.where(element[0] == 0, gennew(element[0]),element[0])
myfunc(array2)

#myarray2 = jnp.array([arr2,b])
#result = jax.vmap(myfunc)([arr1,a],[arr2,b])

#print("result: ",result)
#print("ndim of result: ",result.ndim)
#print("ndim of input:", myarray1.ndim)
#addedcoords = jnp.clip(addedcoords,min=0, max=2)
#[coordinates +1] [coordinates+1] 

