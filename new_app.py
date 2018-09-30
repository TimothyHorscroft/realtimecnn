import cv2
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torchvision as tv
import numpy
import math
import time

from trainer import *


# Initialise training datasets
CLASSES = ( # CLASSES is a tuple of tuples
    tuple(str(i) for i in range(10)), # ("0", "1", ..., "9")
    ("t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"),
    ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
)
img_transforms = tv.transforms.Compose(( # Torchvision conveniently allows the composition of all transforms to be applied to the training images into just one transform
    tv.transforms.ToTensor(), # PyTorch's 'Tensors' are used due to their similarity to numpy arrays, which provide many benefits over multi-dimensional Python lists
                              # The main benefit is the 'shape' attribute, which allows easy debugging of the network, as it is easy to identify the shape after convolutions
                              # Another useful benefit is that printing numpy arrays or PyTorch tensors is much cleaner than lists of lists
    tv.transforms.Lambda(lambda x: 2*x - 1) # the values initially range from 0 to 1; after this transform, they range from -1 to 1
))
# ############### COMMENT ABOUT DATALOADER GOES HERE DONT FORGET AGAIN @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
DATA_ITERATORS = (
    iter(torch.utils.data.DataLoader(
        tv.datasets.MNIST(root="./digit_data", train=True, transform=img_transforms, download=True), # train=True makes the iterator loop indefinitely
        batch_size=1,   # train on one image at a time
        shuffle=True    # randomise the order in which the images appear
    )),
    iter(torch.utils.data.DataLoader(
        tv.datasets.FashionMNIST(root="./fashion_data", train=True, transform=img_transforms, download=True),
        batch_size=1,
        shuffle=True
    )),
    iter(torch.utils.data.DataLoader(
        tv.datasets.CIFAR10(root="./cifar_data", train=True, transform=img_transforms, download=True),
        batch_size=1,
        shuffle=True
    ))
)


class Window:
    def __init__(self, master, width, camera_index=0, max_fps=20):
        self.master = master
        self.width = width

        # Define render constants
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
        self.started = False
        self.cutoff = 40

        # Trainer
        self.trainer = Trainer(DATA_ITERATORS[0], True)

        # Layout (first, since menus use layout variables)
        self.main_canvas = tk.Canvas(master, width=width, height=self.height)
        self.main_canvas.configure(background="#FFFF80")
        self.main_canvas.pack(fill="both")

        # Menus
        self.menu_main = tk.Menu(master, tearoff=0)
        master.config(menu=self.menu_main)

        self.menu_mode = tk.Menu(self.menu_main, tearoff=0)
        self.menu_main.add_cascade(label="Mode", menu=self.menu_mode)
        self.mode = tk.IntVar()
        for value, label in enumerate(("Training", "Video", "Drawing")):
            self.menu_mode.add_radiobutton(label=label+" Mode", variable=self.mode, value=value, command=self.update)

        self.menu_dataset = tk.Menu(self.menu_main, tearoff=0)
        self.menu_main.add_cascade(label="Dataset", menu=self.menu_dataset)
        self.dataset = tk.IntVar()
        for value, label in enumerate(("Digits", "Fashion", "CIFAR-10")):
            self.menu_dataset.add_radiobutton(label=label, variable=self.dataset, value=value, command=self.update)

        self.menu_step = tk.Menu(self.menu_main, tearoff=0)
        self.menu_main.add_cascade(label="Step", menu=self.menu_step)
        self.menu_step.add_command(label="Single", command=self.training_step)

        self.update()

    def training_step(self):
        self.trainer.training_step()
        self.started = True
        self.update()

    def update(self):
        start = time.time()
        self.render_probabilities()

        mode = self.mode.get() # self.mode is a 'tk.IntVar' so it can be modified by menus; this sets 'mode' equal to its int value
        # State Machine
        if mode == 0:
            self.render_dataimage()
        elif mode == 1:
            self.render_video()
        elif mode == 2:
            self.render_drawnimage()

        # Tell the master to call this function again at the correct fps
        if mode == 1:
            self.master.after(1000//self.max_fps, self.update)

        print(f"{time.time() - start:.4f}")

    def render_probabilities(self):
        for i, probs in enumerate(zip(*torch.sort(self.trainer.probabilities[0], descending=True))):
            prob, probIndex = probs # fancy way of iterating such that 'i' goes from 0 to 9, 'probIndex' is the index of the class and 'prob' is the probability of that index
            self.main_canvas.create_rectangle(
                self.padding,
                self.padding + i*(self.box_height+self.padding),
                self.padding + self.box_width,
                (i+1)*(self.box_height+self.padding),
                outline="",
                fill="#E0E0FF"
            )
            self.main_canvas.create_rectangle(
                self.padding,
                self.padding + i*(self.box_height+self.padding),
                self.padding + math.ceil(prob*self.box_width),
                (i+1)*(self.box_height+self.padding),
                outline="",
                fill="#8080FF"
            )

    def render_dataimage(self):
        if self.started:
            self.master.dataimage = dataimage = array_to_photoimage(self.trainer.images[0].numpy(), mode="L")
            self.main_canvas.create_image(
                (self.width - self.image_render_size - self.padding, self.padding),
                image=dataimage,
                anchor="nw"
            )
        else:
            self.main_canvas.create_rectangle(
                self.width - self.image_render_size - self.padding,
                self.padding,
                self.width - self.padding,
                self.padding + self.image_render_size,
                outline="",
                fill="black"
            )

    def render_video(self):
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

        self.master.new_cap = new_cap = array_to_photoimage(frame, new_size=(256, 256))

        # Rendering
        self.main_canvas.create_image(
            (self.width - self.cap_width - 2*self.padding, self.instructions_top + self.padding),
            image=raw_cap,anchor="nw"
        )
        self.main_canvas.create_image(
            (self.width - self.image_render_size - self.padding, self.padding),
            image=new_cap, anchor="nw"
        )

    def render_drawnimage(self):
        pass


def array_to_photoimage(array, mode="RGB", new_size=None):
    image = Image.frombytes(mode, (array.shape[1], array.shape[0]), array)
    if new_size is not None:
        image = image.resize(new_size)
    return ImageTk.PhotoImage(image=image)


root = tk.Tk()
window = Window(root, 1024)
root.mainloop()
