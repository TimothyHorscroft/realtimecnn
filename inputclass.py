import pygame


# A class designed to handle keyboard and mouse inputs, including mouse position
class Input:
    NUM_KEYS = 400 # This is just an estimate, this is fine unless a key has an ID greater than 400
    NUM_BUTTONS = 3 # Left, middle, right

    def __init__(self):
        self.keys = [False] * Input.NUM_KEYS
        self.mouse = [False] * Input.NUM_BUTTONS
        # More arrays and mouse position variables are defined at the beginning of each call to tick()

        self.quit = False

    def reset_keys(self):
        self.keys_p = [False] * Input.NUM_KEYS # only True when a key is first pressed (not when it is held)
        self.keys_r = [False] * Input.NUM_KEYS # only True when a key is released
        self.mouse_p = [False] * Input.NUM_BUTTONS
        self.mouse_r = [False] * Input.NUM_BUTTONS

    def tick(self):
        self.reset_keys()
        self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
                return
            if event.type == pygame.KEYDOWN:
                self.keys[event.key] = True
                self.keys_p[event.key] = True
            elif event.type == pygame.KEYUP:
                self.keys[event.key] = False
                self.keys_r[event.key] = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse[event.button-1] = True # Pygame uses 1,2,3 for the mouse buttons. Since arrays start at 0, 1 is subtracted
                self.mouse_p[event.button-1] = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse[event.button-1] = False
                self.mouse_r[event.button-1] = True
