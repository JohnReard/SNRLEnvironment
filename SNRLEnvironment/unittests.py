import unittest
import jax
import jax.numpy as jnp
import jax.random
#import training, Env1
from training import agent, key, optimizer, train_step
from Env1 import statestep
import time
from batching import create_envbatch
import os
import psutil


class AgentTests(unittest.TestCase):
    def setUp(self):
        #in future when more objects are included in the environment agentindex will be dynamically assigned
        self.agentindex = 1
        self.objnum = 2
        seed = 1901
        key = jax.random.key(seed)
        #make agent move Do i need to call the functions or can i just take the variables straight from the training.py file?
        self.envnum = 3 #test if at minimum feature works for 20 environments
        self.limits = jnp.array([600,600])
        self.prestates = create_envbatch(key,self.envnum,self.limits) #test state for 2 environments
        

        #get agentstates
        self.preagentstates = jax.vmap(lambda x : x[self.agentindex])(self.prestates)

        #call functions, statestep works when there are multiple actions
        loss, self.actions = train_step(agent.policy,self.prestates,optimizer)
        self.poststates = jax.vmap(statestep, in_axes=(0,0,None))(self.prestates,self.actions, self.limits)
        
        #calculate what post states should be, works when there is one action
        self.agentstates_expected = jax.vmap(lambda x, y : x + y,in_axes=(0,0))(self.preagentstates,self.actions)
        self.agentstates_expected=jnp.clip(self.agentstates_expected,min=0,max=self.limits[0]) #needs to be updated if limits aren't equal

    def test_action_effect(self):
        #arrange
        self.setUp()
        #test agent pos has changed
        i = 0
        while i < self.envnum:
            #test agent pos has changed by amount of action
            self.assertTrue(jnp.array_equal(self.agentstates_expected[i],self.poststates[i][self.agentindex]))
            #test agent has moved correct amount?

            #test goal pos has not changed
            self.assertTrue(jnp.array_equal(self.prestates[i][0], self.poststates[i][0]))
            i += 1

        self.tearDown()
    def test_limits(self):
        #arange
        self.setUp()

        xlimit = jnp.array(self.limits.at[0].get())
        ylimit = jnp.array(self.limits.at[1].get())
        agentstates = self.poststates[self.agentindex]
        goalstates = self.poststates[0]
        #assert that states are within limits
        #check agent and goal aren't beyond the x
        self.assertFalse(jnp.any(jnp.greater(agentstates, xlimit)))
        self.assertFalse(jnp.any(jnp.greater(goalstates, xlimit)))
        #check agent and goal aren't beyond the y
        self.assertFalse(jnp.any(jnp.greater(agentstates, ylimit)))
        self.assertFalse(jnp.any(jnp.greater(goalstates, ylimit)))

        self.tearDown()
    def test_statedims(self):
        self.setUp()
        self.assertEqual(jnp.shape(self.prestates),(self.envnum, self.objnum,2))
        self.assertEqual(jnp.shape(self.poststates),jnp.shape(self.prestates))

        self.tearDown()
    #def test_initstates(self):
    #    self.setUp()
#
#        self.tearDown()
class EnvTests(unittest.TestCase):
    def setUp(self):
        a = jnp.zeros(100)
        b = jnp.zeros(10000000)
        def jitfunc(arr):
            jax.vmap(lambda el: el + 1)(arr)
        def timedfunc(arr):
            for el in arr:
                el + 1
        jittedfunc = jax.jit(jitfunc)

        #call jitted functions
        beforea = time.time()
        jittedfunc(a)
        aftera = time.time()

        beforeb = time.time()
        jittedfunc(b)
        afterb = time.time()

        self.jitatime = afterb-beforeb
        self.jitbtime = aftera-beforea

        #call non-jitted functions
        beforea = time.time()
        timedfunc(a)
        aftera = time.time()

        beforeb = time.time()
        timedfunc(b)
        afterb = time.time()

        self.timedatime = afterb-beforeb
        self.timedbtime = aftera-beforea

    def test_paraspeedup(self):
        self.setUp()
        jitimediff = self.jitbtime-self.jitatime
        timeddiff = self.timedbtime-self.timedatime
        print(f'{self.timedbtime:.{7}f}',f'{self.timedatime:.{7}f}')
        self.assertTrue(self.jitbtime<self.jitatime*10)#as f(b) has 10* more operations than f(a), if parallelised time(f(b)) should be less than time(f(a))*10
        self.assertTrue(self.timedbtime>self.timedatime)
        #self.assertEqual(self.timedatime,self.timedbtime)
        


        devices = jax.devices()
        print(devices)
        self.tearDown()
    #def test_coreusage():

if __name__ == '__main__':
    unittest.main()