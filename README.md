"# SNRLEnvironment" 
CHANGELOG:
- Created unittests.py.
- Implemented unit test 1.1
- Stopped returning newcoords in statestep.



FEATURES TO IMPLEMENT:

In unittest.py:
NOW:
- Create Unit test 1.2 and 1.3 (refer to test document).

In training.py:

NOW:
- Create batching for environments, e.g put in a number and have that number of environments engage in training runs.



FUTURE:
- Need to pass input to policy and pass output from policy in agentact func. (output needs to be returned in agentneuralnetwork.py beforehand)


In Env1.py:

FUTURE:

- Limits are currently hardcoded into addvelocity, change this in future so the limits can be set universally.


In agentneuralnetwork.py:

NOW:

- Need to construct Block, Model classes.
- Need to research how to implement layers etc in nnx.



FUTURE:
- Need to pass output from input in policy.

In renderer.py:

NOW: 
- Draw the previous path of the agent, DO NOT INCLUDE THIS PATH AS IMAGE PERCEPT/INPUT.

Misc:

FUTURE:
- Need to allow input to be the image of the environment, i.e the array of all pixels.