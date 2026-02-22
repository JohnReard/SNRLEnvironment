"# SNRLEnvironment" 
CHANGELOG:
- limits are no longer hardcoded, are passed into statestep function.
- coords are now clipped to a minimum of 0 rather than -600.
- unit test 1.2 is now implemented.



FEATURES TO IMPLEMENT:

In unittest.py:
NOW:
- implement unit tests up to 1.5

In training.py:

NOW:
- Create batching for environments, e.g put in a number and have that number of environments engage in training runs.



FUTURE:
- Need to pass input to policy and pass output from policy in agentact func. (output needs to be returned in agentneuralnetwork.py beforehand)


In Env1.py:

FUTURE:

- When limits are clipped they are assumed to be equal (i.e only the y limit is clipped from), make sure both x and y are clipped from.


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