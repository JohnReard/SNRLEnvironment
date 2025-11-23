from Env1 import Environment, State, Agent, Action
from renderer import drawframe, drawwindow

initialstate = State((500,300),(400,300))
agent = Agent(initialstate)
windowwidth = 600
windowheight = 600
#action1 = Action(0.1,0)
#action2 = Action(0,1)
#actionset = [action1,action2]
# env limits must be >= than window size
env = Environment((windowwidth*64,windowheight*64),initialstate,agent, 2)
running = True
i=0

window = drawwindow(windowwidth,windowheight)
while running:
    env.statestep()
    drawframe(env,agent, window)
    i+=1
    if i > 200:
        print("agentposlist:", env.agentposlist)
        print("agenvelocitylist:", env.agenvelocitylist)
        print("agentposlist from agent:", agent.agentposlist)
        running = False
    