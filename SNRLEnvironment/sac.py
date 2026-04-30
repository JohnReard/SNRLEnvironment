from flax import nnx
import jax.numpy as jnp
import jax
import optax

class critic(nnx.Module):
    def __init__(self, features, hidden, targets, rngs : nnx.Rngs):
        self.l1 = nnx.Linear(features, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden,hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden,hidden, rngs=rngs)
        self.l4 = nnx.Linear(hidden,targets, rngs=rngs)
        
    def __call__(self, x):#x in the critic is the state AFTER action is applied
        x = self.l1(nnx.relu(x))
        x = self.l2(nnx.relu(x))
        x = self.l3(nnx.relu(x))
        x = self.l4(nnx.relu(x))
        #maximum entropy objective is the most random action that the agent comes up with.
        #  #x is the value of the action for that state/future states and their actions
        return x
class valuenetwork(nnx.Module):
    def __init__(self, features, hidden, targets, rngs : nnx.Rngs):
        self.l1 = nnx.Linear(features, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden,hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden,hidden, rngs=rngs)
        self.l4 = nnx.Linear(hidden,targets, rngs=rngs)
    def __call__(self, x):
        x = self.l1(nnx.relu(x))
        x = self.l2(nnx.relu(x))
        x = self.l3(nnx.relu(x))
        x = self.l4(nnx.relu(x))
        return x

    
class actor(nnx.Module):
    def __init__(self, features, hidden, targets, rngs : nnx.Rngs):
        self.l1 = nnx.Linear(features, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden,hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden,hidden, rngs=rngs)
        self.mean = nnx.Linear(hidden,targets, rngs=rngs)
        self.variance = nnx.Linear(hidden,targets, rngs=rngs)
        self.rngs = rngs
    def __call__(self, x):
        x = self.l1(nnx.relu(x))
        x = self.l2(nnx.relu(x))
        mean = self.mean(x)
        variance = self.variance(nnx.sigmoid(x))
        #clip variance?
        return mean, variance
        #maximum entropy objective is the most random action that the agent comes up with.
    def samplenormal(self,x):#necessary?
        mean, variance = self(x)
        probability_distribution = jax.random.normal(mean, variance)
        #what is reparameterise?
        rngs = jax.random.split(self.rngs)
        actions = jax.random.choice(rngs,probability_distribution)

        action = nnx.tanh(actions)
        
        return action, probability_distribution

