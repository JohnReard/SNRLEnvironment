import pygame

pygame.init()
def drawwindow(windowheight, windowwidth):
    window = pygame.display.set_mode((windowwidth, windowheight))
    window.fill((255, 255, 255))
    return window

def drawframe(env, agent, window):
    window.fill((255, 255, 255))
    #draw agent
    print("agentpos",env.currentstate.agentpos[0], env.currentstate.agentpos[1])
    #has to be drawn as cartesian, will be polar in Env1 and others.
    pygame.draw.rect(window, (0, 0, 255), [float(env.currentstate.agentpos[0])/100, float(env.currentstate.agentpos[1])/100, 20, 20], 0)
    #draw goal
    pygame.draw.rect(window, (0, 255, 0), [env.currentstate.goalpos[0],env.currentstate.goalpos[1], 30, 30], 0)
    pygame.display.update()
    #agent is blue goal is green
    #if event.type == pygame.QUIT:
    #   running = False

    #pygame.quit()