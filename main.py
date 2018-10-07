from appclass import *


app = App(1024)

while True:
    app.tick()
    if app.inp.quit:
        app.destroy()
        break
    app.render()

    pygame.display.update()
