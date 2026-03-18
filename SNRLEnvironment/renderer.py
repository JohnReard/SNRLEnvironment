import tensorcanvas as tc
import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib.animation as animation



def drawwindow(height, width):
    window = jnp.zeros([height, width, 3]) #what is channels doing?
    return window

def showplt(window):
    #window = jnp.permute_dims(window, (1,0,2))#changes dimensions of array to valid dims for show()
    #plt.plot(window)
    window = jnp.array(window)
    plt.imshow(window)
    plt.show()
def drawshape(window,state,colour):
    x = state[0].astype(int)
    y = state[1].astype(int)
    radius = state[2].astype(int)
    window = tc.draw_circle(x,y,radius,colour,window)
    return window
def createplot():
    fig, ax = plt.subplots()
def drawframe(states, window,collision):
    i=0
    for state in states:
        if i == 0:
            colour = jnp.array([0.1,0.8,0.2])
        elif i == 1:
            colour = jnp.array([0.1,0.1,0.8])
        #elif i == 1 and collision>0:
        #    colour = jnp.array([0.7,0.1,0.5])
        else:
            colour = jnp.array([0.8,0.1,0.2])
        window = drawshape(window,state,colour)
        i+=1
    return window
    #plt.tight_layout()
    
def update(frames,i):
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
