import unittest
import jax
import jax.numpy as jnp
#import training, Env1
from training import pureact, key, subkey
from Env1 import statestep

class AgentTests(unittest.TestCase):
    def setUp(self):
        #make agent move Do i need to call the functions or can i just take the variables straight from the training.py file?
        self.envnum = 2 #test if at minimum feature works for 2 environments
        prestates = jnp.array([[[200,200],[300,300]],[[400,400],[500,500]]]) #test state for 2 environments
        
        #call functions
        agentact = jax.vmap(pureact,in_axes=(self.envnum,None, None))
        self.actions = agentact(jnp.array([prestates]), key, subkey)
        poststates = jax.vmap(statestep)(prestates,self.actions)

        #extract agent states 
        preagentstate_env1 = prestates[0][1]
        preagentstate_env2 = prestates[1][1]
        postagentstate_env1 = poststates[0][1]
        postagentsate_env2 = poststates[1][1]
        
        self.preagentstates = jnp.array([preagentstate_env1,preagentstate_env2])
        self.postagentstates = jnp.array([postagentstate_env1,postagentsate_env2])
        
        #calculate what post states should be
        self.agentstates_expected = jax.vmap(lambda x, y : x + y)(self.preagentstates,self.actions)

        #extract goal states
        pregoalstate_env1 = prestates[0][0]
        pregoalstate_env2 = prestates[1][0]
        postgoalstate_env1 = poststates[0][0]
        postgoalstate_env2 = poststates[1][0]

        self.pregoalstates = [pregoalstate_env1,pregoalstate_env2]
        self.postgoalstates = [postgoalstate_env1, postgoalstate_env2]




    def test_action_effect(self):
        #arrange
        self.setUp()
        #test agent pos has changed
        i = 0
        while i < self.envnum:
            #test agent pos has changed by amount of action
            self.assertTrue(jnp.array_equal(self.agentstates_expected[i],self.postagentstates[i]))
            #test agent has moved correct amount?\

            #test goal pos has not changed
            self.assertTrue(jnp.array_equal(self.pregoalstates[i], self.postgoalstates[i]))
            i += 1

        self.tearDown()
if __name__ == '__main__':
    unittest.main()