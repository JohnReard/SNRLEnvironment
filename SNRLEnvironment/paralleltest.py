import jax
import jax.numpy as jnp

devicecount = jax.device_count()
devices = jax.devices()


def addsclrs(x,y):
    return x+y

input1 = jnp.array([10,2,2,2])
input2 = jnp.array([10,2,2,2])
input1=jnp.reshape(input1,(2,2))
input2=jnp.reshape(input2,(2,2))
print("Dimensions:",input1.ndim, input2.ndim,"Shapes:",input1.shape, input2.shape)
print("input1:", input1.shape)
print("input2:", input2.shape)
print("Devices: ", devices)

print(jax.pmap(addsclrs)(input1,input2))