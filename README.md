"# SNRLEnvironment" 

FEATURES TO IMPLEMENT:

In training.py:

NOW:

- Need to check if agent's state is beyond limits before updating the state.
FUTURE:
- Need to pass input to policy and pass output from policy in agentact func. (output needs to be returned in agentneuralnetwork.py beforehand)


In Env1.py:

- Limits are currently hardcoded into addvelocity, change this in future so the limits can be set universally.


In agentneuralnetwork.py:

NOW:

- Need to construct Block, Model classes.
- Need to research how to implement layers etc in nnx.

FUTURE:
- Need to pass output from input in policy.