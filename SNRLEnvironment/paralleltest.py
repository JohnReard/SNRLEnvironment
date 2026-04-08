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


myarray = jnp.array([0, 2, 5, 0])
myint = 10
print(myint-jnp.sum(myarray)) 
#myarray1 = jnp.array([arr1,a])
#myarray2 = jnp.array([arr2,b])
#result = jax.vmap(myfunc)([arr1,a],[arr2,b])

#print("result: ",result)
#print("ndim of result: ",result.ndim)
#print("ndim of input:", myarray1.ndim)
#addedcoords = jnp.clip(addedcoords,min=0, max=2)
#[coordinates +1] [coordinates+1] 

