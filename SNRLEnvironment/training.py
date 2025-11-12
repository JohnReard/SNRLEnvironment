from Env1 import Environment, State, Agent, Action
from renderer import drawframe, drawwindow

initialstate = State((500,300),(0,0))
agent = Agent()
action1 = Action(0.1,0)
action2 = Action(0,1)
actionset = [action1,action2]
# env limits must be >= than window size
env = Environment((600,600),actionset,initialstate,agent)
running = True
i=0

window = drawwindow()
while running:
    env.statestep()
    drawframe(env,agent, window)
    i+=1
    if i > 200:
        running = False
    