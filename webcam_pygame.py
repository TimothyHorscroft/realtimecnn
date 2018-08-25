import cv2
import pygame
import numpy

pygame.init()

vidcap = cv2.VideoCapture(0) # 0 is the camera index, modifying it changes which camera is used for video capture

"""
    Annoyingly, the camera resolution can only be set to a few options, so setting the width followed by
    getting the width will often return different values. Thus, when the width is 'set', values like 1 can
    be entered without risk of the camera resolution dropping to 1x1; it will simply pick the lowest
    possible resolution. This code was written on a laptop with a default resolution of 640x480.

    The Pygame window is set to be the same size as the video capture.
"""
#vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1)
WIDTH = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((WIDTH, HEIGHT))


colour_state = 0
running = True
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        if event.type == pygame.KEYDOWN:
            # Use the 'z' and 'x' keys to rotate through each colour scheme
            if event.key == pygame.K_z:
                colour_state = (colour_state - 1) % 4
            elif event.key == pygame.K_x:
                colour_state = (colour_state + 1) % 4
    if not running:
        break
    
    frame = vidcap.read()[1] # the first returned variable indicates success or failure

    if colour_state == 0:
        # traditional
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif colour_state == 1:
        # grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # gray means 'gray BGR', doing this makes the format 'gray RGB'
    # if colour_state == 2 then make no modifications, resulting in the switching of the red and blue colour channels
    elif colour_state == 3:
        # inverted (WARNING: EXTREMELY SLOW!!)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                for c in range(frame.shape[2]):
                    frame[y][x][c] = 255 - frame[y][x][c]

    screen.blit(pygame.surfarray.make_surface(numpy.rot90(frame)), (0, 0))
    
    pygame.display.update()


pygame.quit()
cv2.destroyAllWindows()
