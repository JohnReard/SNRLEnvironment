import tensorcanvas as tc
import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib.animation as animation


def drawwindow(height, width):
    window = jnp.zeros([height, width, 3]) #what is channels doing?
    return window

def showplt(window):
    print("window shape: ", window.shape)
    #window = jnp.permute_dims(window, (1,0,2))#changes dimensions of array to valid dims for show()
    print("window type: ", type(window))
    #plt.plot(window)
    window = jnp.array(window)
    print(window)
    plt.imshow(window)
    plt.show()

def drawframe(goalvel, agentvel, window,i):
    #draw agent
    agent_radius = 7.0
    agent_colour = jnp.array([0.1,0.1,0.8])
    #polar coordinates??
    ax = int(agentvel[0])
    ay = int(agentvel[1])
    window = tc.draw_circle(ax,ay,agent_radius,agent_colour,window)
    #draw goal
    goal_radius = 10.0 + i
    goal_colour = jnp.array([0.1,0.8,0.2])
    window = tc.draw_circle(goalvel[0],goalvel[1],goal_radius,goal_colour,window)
    print("Window shape:", window.shape)
    return window
    #plt.tight_layout()
    
def update(frames,i ):
    x=(frames[i][0])
    y=(frames[i][1])
    #on the first frame draw the window and image
    #if i == 0:
    #    ax.imshow(frames[i], animated=True)
    #on every(maybe else instead?) frame draw image
    #anim.append(frames[i])
    #graph.set_data(x, y)
    plt.plot(x,y)
    i+=1
    
    #agent is blue goal is green