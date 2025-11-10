import pygame

pygame.init()
def drawwindow():
    window = pygame.display.set_mode((600, 600))
    window.fill((255, 255, 255))
    return window

def drawframe(env, agent, window):
    #draw agent
    pygame.draw.rect(window, (0, 0, 255), [env.currentstate.agentpos[0], env.currentstate.agentpos[1], 70, 70], 0)
    #draw goal
    pygame.draw.rect(window, (0, 255, 0), [env.currentstate.goalpos[0],env.currentstate.goalpos[1], 70, 70], 0)
    pygame.display.update()
    #if event.type == pygame.QUIT:
    #   running = False

    #pygame.quit()