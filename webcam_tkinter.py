import cv2
import tkinter as tk
from PIL import Image, ImageTk


class Capture:
    def __init__(self, master, camera_index=0, max_fps=20):
        self.master = master
        self.max_fps = 20

        self.vidcap = cv2.VideoCapture(camera_index)

        """
            Annoyingly, the camera resolution can only be set to a few options, so setting the width followed by
            getting the width will often return different values. Thus, when the width is 'set', values like 1 can
            be entered without risk of the camera resolution dropping to 1x1; it will simply pick the lowest
            possible resolution. This code was written on a laptop with a default resolution of 640x480.
        """
        #self.vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1)
        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = tk.Canvas(master, width=self.width, height=self.height)
        self.canvas.pack()

        self.colour_state = tk.IntVar()

        self.update() # after this function is called once, it is called many times per second until the window closes

    def update(self):
        frame = self.vidcap.read()[1] # the first returned variable indicates success or failure
        colour_state = self.colour_state.get()

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

        frame = Image.frombytes('RGB', (self.width, self.height), frame)
        frame = ImageTk.PhotoImage(image=frame)
        self.master.frame = frame # prevent 'frame' from being garbage collected

        self.canvas.create_image(0, 0, image=frame, anchor="nw")

        # Tell the master to call this function again after 50ms
        self.master.after(1000//self.max_fps, self.update)


root = tk.Tk()
capture = Capture(root)

menu_main = tk.Menu(root, tearoff=0)
root.config(menu=menu_main)

menu_colour = tk.Menu(menu_main, tearoff=0)
menu_main.add_cascade(label="Colour", menu=menu_colour)
for value, label in enumerate(("RGB", "Grayscale", "BGR", "Inverted (WARNING: slow)")):
    menu_colour.add_radiobutton(label=label, variable=capture.colour_state, value=value)

root.mainloop()

cv2.destroyAllWindows()
