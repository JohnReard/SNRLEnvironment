import jax
import jax.numpy as jnp
import os
import psutil
import subprocess
#os.environ['CUDA_VISIBLE_DEVICES'] = '20'
print(psutil.cpu_count())
print(psutil.Process().cpu_affinity())
print(psutil.Process().cpu_affinity())
print(psutil.cpu_freq(percpu=True))
array = jnp.array([0,10])
print(jnp.append(array, 19))
@jax.jit
def createarray(a):
    return jnp.array([1,1,1])
partitions = jnp.array(jax.vmap(createarray)(jnp.empty(31)))
#partitions[0] = array

partitions = partitions.at[0].set([19,9,2])
print(partitions)
myarray = jnp.array([3,4,22])
newpartition = jnp.append(partitions[0],myarray)
partitions = jnp.append(partitions[0],newpartition)
print(partitions)
#print(subprocess.run(["powershell", "-Command", "Get-Counter '\\Processor(_Total)\\% Processor Time' -MaxSamples 1"],capture_output=True))