from appclass import *

app = App(1024)             # 1024 is the width of the Application window

while True:
    app.tick()
    if app.inp.quit:        # This is 'True' only if the window is closed
        app.destroy()       # This calls pygame.quit()
        break               # Exit the entire program
    app.render()
    pygame.display.update() # Update the screen
