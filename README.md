"# SNRLEnvironment" 
FEATURES TO IMPLEMENT:
In training.py:
NOW:
- Need to check if agent's state is beyond limits before updating the state.
- Create a function to batch environments rather than batching manually.
FUTURE:
- Need to pass input to policy and pass output from policy in agentact func. (output needs to be returned in agentneuralnetwork.py beforehand)
In Env1.py:


In agentneuralnetwork.py:
- Need to construct Block, Model classes.
- Need to research how to implement layers etc in nnx.
FUTURE:
- Need to pass output from input in policy.

In renderer.py:
- Create a visualiser that creates visualisation for one environment, maybe use GPU to render it?