from Env1 import Environment, State, Agent, Action, statestep, addvelocity
from renderer import drawframe, drawwindow
import jax
import jax.numpy as jnp
initialstate = State((500,300),(400,300))
agent = Agent(initialstate)
agent2 = Agent(initialstate)
windowwidth = 600
windowheight = 600
#action1 = Action(0.1,0)
#action2 = Action(0,1)
#actionset = [action1,action2]
# env limits must be >= than window size
env = Environment((windowwidth*64,windowheight*64),initialstate,agent,2)
env2 = Environment((windowwidth*64,windowheight*64),initialstate,agent,2)
#environments = jnp.array(env,env2)
running = True
i=0



window = drawwindow(windowwidth,windowheight)
window2 = drawwindow(windowwidth,windowheight)
while running:
    jax.tree.map(statestep, [env, env2])
    #env.currentstate = env.statestep()

    #env2.statestep()
    drawframe(env,agent, window)
    drawframe(env2,agent2, window2)
    i+=1
    if i > 200:
        print("agentposlist:", env.agentposlist)
        print("agenvelocitylist:", env.agenvelocitylist)
        print("agentposlist from agent:", agent.agentposlist)
        running = False
    