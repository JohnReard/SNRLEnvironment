"# SNRLEnvironment" 

FEATURES TO IMPLEMENT:

In training.py:

NOW:
- Bug with agent being drawn twice.

FUTURE:
- Need to pass input to policy and pass output from policy in agentact func. (output needs to be returned in agentneuralnetwork.py beforehand)


In Env1.py:

- Limits are currently hardcoded into addvelocity, change this in future so the limits can be set universally.


In agentneuralnetwork.py:

NOW:

- Need to construct Block, Model classes.
- Need to research how to implement layers etc in nnx.
- Need to reimplement random in jax.random rather than PNRG (not GPU compatible and I think deprecated?)


FUTURE:
- Need to pass output from input in policy.
