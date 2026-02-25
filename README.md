"# SNRLEnvironment" 
CHANGELOG:
- environments now created by batching.py.
- batching implemented for custom environment size
- unittests 1.1, 1.12 updated for batching



FEATURES TO IMPLEMENT:

In unittest.py:
NOW:
- implement unit tests up to 1.5
- new unit test for length of environments?

In training.py:

NOW:
- Images need to be drawn for all environments, drawframe needs to be jitted for this.



FUTURE:
- Need to pass input to policy and pass output from policy in agentact func. (output needs to be returned in agentneuralnetwork.py beforehand)


In Env1.py:

FUTURE:

- When limits are clipped they are assumed to be equal (i.e only the y limit is clipped from), make sure both x and y are clipped from.
- When more object states are added to the environment statestep will have to find a way to get the agentstate no matter what the size of the array is.


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