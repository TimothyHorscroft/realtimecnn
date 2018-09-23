import cv2
import tkinter as tk
from PIL import Image, ImageTk
import torch


class Window:
    def __init__(self, master, width, camera_index=0, max_fps=20):
        self.master = master
        self.width = width

        # Define render constants (see user manual for how to change dimensions)
        self.image_render_size = width // 4 # all variables are defined in terms of width, including the height
        self.padding = width // 128
        self.font_size = width // 64
        self.box_height = 2*self.padding + self.font_size
        self.box_width = width - 3*self.padding - self.image_render_size
        self.instructions_top = max(10*self.box_height, 11*self.padding, self.image_render_size + 5*self.font_size + 9*self.padding) # the max function is used to be safe, as it is unclear which column of the app is lower, and this may vary with height
        self.height = self.instructions_top + 10*self.font_size + 12*self.padding

        # Update Master
        master.resizable(False, False)
        master.geometry(f"{width}x{self.height}")
        master.update()

        # Set up camera
        self.capture = cv2.VideoCapture(camera_index)
        self.max_fps = max_fps

        """
            Annoyingly, the camera resolution can only be set to a few options, so setting the width followed by
            getting the width will often return different values. Thus, when the width is 'set', values like 1 can
            be entered without risk of the camera resolution dropping to 1x1; it will simply pick the lowest
            possible resolution. This code was written on a laptop with a default resolution of 640x480.
        """
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1)
        self.cap_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Misc variables
        self.cutoff = 40

        # Layout (first, since menus use layout variables)
        self.main_canvas = tk.Canvas(master, width=width, height=self.height)
        self.main_canvas.configure(background="#FFFF80")
        self.main_canvas.pack(fill="both")

        # Menus
        self.menu_main = tk.Menu(master, tearoff=0)
        master.config(menu=self.menu_main)

        self.menu_colour = tk.Menu(self.menu_main, tearoff=0)
        self.menu_main.add_cascade(label="Options", menu=self.menu_colour)
        self.dataset = tk.IntVar()
        for value, label in enumerate(("Digits", "Fashion", "CIFAR-10")):
            self.menu_colour.add_radiobutton(label=label, variable=self.dataset, value=value)

        self.update() # after this function is called once, it is called many times per second until the window closes

    def update(self):
        frame = self.capture.read()[1]
        dataset = self.dataset.get()

        # The image is landscape format, so cut it to make it square
        for row in frame:
            row = row[:self.cap_height]

        if dataset != 2:
            # GRAYSCALE
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # gray means 'gray BGR', doing this makes the format 'gray RGB'
            self.master.raw_cap = raw_cap = array_to_photoimage(frame) # Save a copy of the raw frame before further modification
            # the above trick is to prevent garbage collection
            frame = cv2.resize(frame, dsize=(28, 28))
            images = torch.empty(1, 1, 28, 28)
            for i in range(28):
                for j in range(28):
                    if frame[i][j][0] < self.cutoff:
                        images[0][0][i][j] = -1
                        for c in range(3):
                            frame[i][j][c] = 0
                    else:
                        images[0][0][i][j] = 1
                        for c in range(3):
                            frame[i][j][c] = 255
        else:
            # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.master.raw_cap = raw_cap = array_to_photoimage(frame)
            frame = cv2.resize(frame, dsize=(32, 32))
            images = torch.empty(1, 3, 32, 32)
            for i in range(32):
                for j in range(32):
                    for c in range(3):
                        images[0][c][i][j] = frame[i][j][c]/128 - 1

        self.master.new_cap = new_cap = array_to_photoimage(frame, (256, 256))

        self.main_canvas.create_image((10, 10), image=raw_cap, anchor="nw")
        self.main_canvas.create_image((250, 250), image=new_cap, anchor="nw")

        # Tell the master to call this function again at the correct fps
        self.master.after(1000//self.max_fps, self.update)


def array_to_photoimage(array, new_size=None):
    image = Image.frombytes("RGB", (array.shape[1], array.shape[0]), array)
    if new_size is not None:
        image = image.resize(new_size)
    return ImageTk.PhotoImage(image=image)


root = tk.Tk()
window = Window(root, 1024)
root.mainloop()
