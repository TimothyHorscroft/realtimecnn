import pygame
import cv2
import torch
import torchvision as tv

from trainer import *       # Splitting up the program into different files makes it easier to find each piece
from inputclass import *    # Python has an in-built 'input' function, so that name is unavailable
from gui import *
from utils import *

import numpy
import math


class App:
    # Datasets
    DIGIT = 0
    FASHION = 1
    CIFAR = 2

    # Modes
    TRAINING = 0
    VIDEO = 1
    DRAWING = 2

    # Video Modes
    RAW = 0
    INVERT = 1
    CUTOFF = 2

    def __init__(self, width):
        self.width = width

        self.load_datasets()
        self.init_pygame()
        self.init_app()

    def load_datasets(self):
        self.dataset = App.DIGIT        # Dataset is changed in the drop-down menu

        # Torchvision conveniently allows the composition of all transforms to be applied to the training images into just one transform
        img_transforms = tv.transforms.Compose((
            tv.transforms.ToTensor(),   # PyTorch's 'Tensors' are used due to their similarity to numpy arrays, which provide many benefits over multi-dimensional Python lists
                                        # The main benefit is the 'shape' attribute, which allows easy debugging of the network, as it is easy to identify the shape after convolutions
                                        # Another useful benefit is that printing numpy arrays or PyTorch tensors is much cleaner than lists of lists
            tv.transforms.Lambda(lambda x: 2*x - 1) # The values initially range from 0 to 1; after this transform, they range from -1 to 1
        ))
        # Define class arrays, dataloader arrays, and 'Trainer' object arrays, as tuples of length 3, one for each dataset
        # The first item in each tuple is displayed as the label, the others are other accepted names when inputting the correct answer for a drawing or video
        self.CLASSES = (
            (("0", "zero"), ("1", "one"), ("2", "two"), ("3", "three"), ("4", "four"), ("5", "five"), ("6", "six"), ("7", "seven"), ("8", "eight"), ("9", "nine")),
            (("t-shirt", "tshirt", "t shirt"), ("trouser", "trousers", "pants"), ("pullover",), ("dress",), ("coat",), ("sandal",), ("shirt",), ("sneaker", "shoe"), ("bag",), ("ankle boot",)),
            (("airplane", "aeroplane", "plane"), ("automobile", "car"), ("bird",), ("cat",), ("deer",), ("dog",), ("frog",), ("horse",), ("ship", "boat"), ("truck",))
        )
        self.DATA_ITERATORS = (
            iter(torch.utils.data.DataLoader(
                tv.datasets.MNIST(root="./digit_data", train=True, transform=img_transforms, download=True), # train=True makes the iterator loop indefinitely
                batch_size=1,           # Train on one image at a time
                shuffle=True            # Randomise the order in which the images appear
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
        self.trainers = tuple(Trainer(self.DATA_ITERATORS[i], i!=2) for i in range(3))  # Create a tuple of 3 Trainers, one for each dataset
        # Note that i!=2 determines whether the dataset is grayscale

    def grayscale(self):
        return self.dataset != 2

    def trainer(self):  # Get the trainer for the current dataset
        return self.trainers[self.dataset]

    def init_pygame(self):
        pygame.init()   # This is necessary to initialise fonts

        self.set_render_constants()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.set_caption()
        self.font = pygame.font.SysFont("monospace", self.font_size)

    def set_render_constants(self):
        self.padding = self.width // 128                    # All variables are defined in terms of width, including the height
        self.font_size = self.width // 64
        self.menu_height = self.font_size + 2*self.padding
        self.image_render_size = self.width // 4
        self.prob_width = self.width - self.image_render_size - 3*self.padding
        self.prob_height = self.font_size + 2*self.padding  # Same as menu_height, but 2 variables exist to avoid confusion
        self.image_left = self.width - self.image_render_size - self.padding
        self.info_top = self.menu_height + self.image_render_size + self.font_size + 5*self.padding
        self.height = max(
            self.menu_height + 10*self.prob_height + 11*self.padding,
            self.info_top + 4*self.font_size + 4*self.padding
        ) # The max function is used to be safe, as it is unclear which column is lower, and this may vary with height

    def set_caption(self, drawingname=""):
        if drawingname:
            pygame.display.set_caption(f"Real-Time CNN Image Classifier Application ({drawingname})")
        else:
            pygame.display.set_caption("Real-Time CNN Image Classifier Application")

    def init_app(self):
        self.init_popups()
        self.init_modes()
        self.init_brush()
        self.init_capture()
        self.init_training_variables()
        self.init_menu()

        self.inp = Input()                                  # This class handles keyboard and mouse input

    def init_popups(self):
        self.queued = []
        self.pause = False
        self.can_quicksave = False                          # True when a file is opened, so the user does not need to enter the filename each time

        # Initialise the rectangle which covers the screen when the app is paused, to increase the program's speed
        self.transp_rect = pygame.Surface((self.width, self.height))
        self.transp_rect.set_alpha(128)
        self.transp_rect.fill((0, 0, 0))

        self.cutoff = Entry(self, "Set Cutoff Value:", 256, initial_value=40, num_mode=True)    # Any pixel darker will become black, any any pixel brighter will become white
        self.networkname = Entry(self, "Enter Network Name (case sensitive):", 256, initial_value="my_network")
        self.drawingname = Entry(self, "Enter Drawing Name (case sensitive):", 256, initial_value="my_drawing")
        self.answername = Entry(self, "Enter Answer (case insensitive):", 256, initial_value="cat")
        self.msg = Popup(self)                              # The same object can be used for all messages, as its label can be changed

        self.popups = (self.cutoff, self.networkname, self.drawingname, self.answername, self.msg)

    def init_modes(self):
        self.render_images = [pygame.Surface((28, 28)) for _ in range(3)]   # The image rendered in the black box in the top-right
        self.mode = App.TRAINING
        self.video_mode = App.RAW
        self.rapid_training = False

        self.clear_drawn_image()

    def clear_drawn_image(self):
        self.drawn_images = torch.full((1, 1, 28, 28), -1)  # Network accepts images from -1 to 1
        self.render_images[App.DRAWING].fill((0, 0, 0))     # pygame accepts images from 0 to 255; both images are filled with black

    def init_brush(self):
        # Define the brush shape for drawing (xpos, ypos, colour) where colour==1 -> white and colour==0 -> gray
        self.brush = [(i, j, 1) for i in (0, 1) for j in (0, 1)]
        self.brush.extend((i, j, 0) for i in (0, 1) for j in (-1, 2))
        self.brush.extend((i, j, 0) for i in (-1, 2) for j in (0, 1))

    def init_capture(self):
        self.capture = cv2.VideoCapture(0)                  # 0 is the camera index, modifying it changes which camera is used for video capture
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1)       # Set the resolution of the webcam as low as possible so that iterating through each frame is faster
        self.capture.read()                                 # This is laggy the first time, so make the first time part of the initialisation

    def init_training_variables(self):
        # Initialise correct answers
        self.answer_video = None
        self.answer_drawing = None

    def init_menu(self):
        # Explicitly define the menus which are referenced later
        self.menu_cifar = Menu(self, "CIFAR-10", command=lambda:self.set_dataset(App.CIFAR), hover_msg="Cannot be in drawing mode")
        self.menu_drawing = Menu(self, "Drawing Mode", command=lambda:self.set_mode(App.DRAWING), hover_msg="Dataset must be grayscale")
        self.menu_invert = Menu(self, "Invert", command=lambda:self.set_videomode(App.INVERT), hover_msg="Dataset must be grayscale")
        self.menu_cutoff = Menu(self, "Cutoff", command=lambda:self.set_videomode(App.CUTOFF), hover_msg="Dataset must be grayscale")
        self.cascmenus = (
            Menu(self, "Training", top=True, children=(
                Menu(self, "Execute Training Steps", children=(
                    Menu(self, "1 Step", command=self.training_step),
                ) + tuple(
                    Menu(self, f"{i} Steps", command=self.multi_train_decorator(i)) for i in (50, 100, 500, 1000, 5000, 10000, 50000)
                )),
                Menu(self, "Toggle Rapid Training", command=self.toggle_rapid_training),
                Menu(self, "Step Until Wrong", command=lambda:self.step_until_cond(False)),
                Menu(self, "Step Until Right", command=lambda:self.step_until_cond(True))
            )),
            Menu(self, "Video", top=True, children=(
                Menu(self, "Change Video Mode", children=(
                    Menu(self, "Raw", command=lambda:self.set_videomode(App.RAW)),
                    self.menu_invert,
                    self.menu_cutoff,
                )),
                Menu(self, "Set Correct Answer", command=self.set_answer)
            )),
            Menu(self, "Drawing", top=True, children=(
                Menu(self, "Clear Image", command=self.clear_drawn_image),
                Menu(self, "Set Correct Answer", command=self.set_answer)
            ))
        )

        # Define the recursive GUI structure as a list of Menu items, some of which have a tuple of Menu items as children
        self.menu_items = [
            Menu(self, "File", top=True, children=(
                Menu(self, "New Drawing", command=self.new_drawing),
                Menu(self, "Open Drawing", command=self.open_drawing),
                Menu(self, "Save Drawing", command=self.save_drawing),
                Menu(self, "Save Drawing As", command=self.save_drawing_as),
                Menu(self, "Exit App", command=self.quit)
            )),
            Menu(self, "General Settings", top=True, children=(
                Menu(self, "Mode", children=(
                    Menu(self, "Training Mode", command=lambda:self.set_mode(App.TRAINING)),
                    Menu(self, "Video Mode", command=lambda:self.set_mode(App.VIDEO)),
                    self.menu_drawing
                )),
                Menu(self, "Dataset", children=(
                    Menu(self, "MNIST Digits", command=lambda:self.set_dataset(App.DIGIT)),
                    Menu(self, "MNIST Fashion", command=lambda:self.set_dataset(App.FASHION)),
                    self.menu_cifar
                )),
                Menu(self, "Cutoff", command=self.set_cutoff)
            )),
            Menu(self, "Edit", top=True, children=(
                Menu(self, "Reset Accuracy", command=lambda:self.trainer().reset_accuracy()),   # This must be inside a lambda so that self.trainer() is called each time
            )),                                                                                 # Otherwise, you could do 'command=self.trainer().reset_accuracy'
            self.cascmenus[0]
        ]

        # Place the top-level menu items horizontally across the top of the screen
        x = self.padding
        for menu_item in self.menu_items:
            menu_item.set_pos(x, 0) # y already is 0

            # Now that the positions of the top-level menu items are calculated, calculate the positions of the rest of the recursive structure
            if menu_item.has_children():
                menu_item.init_children()
            x += menu_item.w + 2*self.padding

        for menu_item in self.cascmenus[1:]: # Initialise the menus which are not shown
            menu_item.set_pos(self.cascmenus[0].x, self.cascmenus[0].y)
            menu_item.init_children()

    # The following methods are necessary as they are passed as commands into the GUI Menu structure
    def set_mode(self, mode):
        if self.mode == mode: # If the app is already in this mode, do not bother with the rest of the calculations
            return
        self.mode = mode
        self.rapid_training = False
        self.trainer().started = False
        self.menu_cifar.disabled = (mode == App.DRAWING)    # Disable the CIFAR menu item if 'drawing mode' is switched to
        self.menu_items[-1] = self.cascmenus[mode]          # Change the rightmost menu item based on the new mode

    def set_dataset(self, dataset):
        if self.dataset == dataset:
            return
        self.dataset = dataset
        self.rapid_training = False
        if dataset == App.CIFAR:
            self.video_mode = App.RAW                       # Automatically change to 'raw video' if the 'CIFAR' database is chosen
        for menu in (self.menu_drawing, self.menu_invert, self.menu_cutoff):
            menu.disabled = (dataset == App.CIFAR)          # Disable three menus if the 'CIFAR' database is chosen

    def set_videomode(self, mode):
        self.video_mode = mode

    def pause_app(self):
        self.pause = True
        self.screen.blit(self.transp_rect, (0, 0))          # This rectangle only needs to be blit once, since the rest of the screen does not render when the app is paused

    def set_cutoff(self):
        # Pause the app and allow the user to type into the 'self.cutoff' Entry Class from 'gui.py'
        self.pause_app()
        self.cutoff.active = True

    def prepare_training_image(self):
        # Reformat the tensor/image so that pygame.surfarray.make_surface() accepts it
        # 'tensor_to_image' is slow, so this one of the slowest functions
        # This is an explicit method to reduce the number of times it's called, especially after multiple training steps are executed in a row
        self.render_images[App.TRAINING] = pygame.surfarray.make_surface(
            tensor_to_image(self.trainer().images[0], self.grayscale())
        )

    def training_step(self):
        # Because of the previous decision, a method should be defined which calls a training step AND THEN prepares the image for rendering
        self.trainer().training_step()
        self.prepare_training_image()

    # An explanation for this is included at 'realtimecnn.blogspot.com' in the post 'Menus are harder than they look'
    def multi_train_decorator(self, num):
        def multi_train():
            for _ in range(num):
                self.trainer().training_step()  # Execute training steps without preparing the image for rendering, for efficiency
            self.prepare_training_image()       # After the number of training steps, prepare the image
        return multi_train

    def toggle_rapid_training(self):
        self.rapid_training = not self.rapid_training

    def step_until_cond(self, break_cond):
        self.rapid_training = False             # Cancel rapid training when this menu item is clicked, as otherwise, the method is pointless
        for i in range(1000):                   # If the network achieves sufficiently high accuracy, or memorises the data, cap this at 1000 steps
            self.trainer().training_step()      # Otherwise (e.g. using a while loop) an infinite loop may occur
            if self.trainer().correct_guess == break_cond:
                self.prepare_training_image()
                return

    def set_answer(self):
        self.pause_app()
        self.answername.active = True
        self.queued.append(self.set_answer_2)   # Rather than accept the text entry, it must be checked afterwards, in this function

    def set_answer_2(self):
        answer = self.answername.get().lower()
        for index, item in enumerate(self.CLASSES[self.dataset]):
            if answer in item:                  # For each classification, check if the text is in the tuple of acceptable answers
                if self.mode == App.VIDEO:
                    self.answer_video = index
                elif self.mode == App.DRAWING:
                    self.answer_drawing = index
                return

        # Entered answer was not one of the acceptable classes
        self.pause_app()
        self.msg.set_label("Invalid Answer")
        self.msg.active = True

    def new_drawing(self):
        self.can_quicksave = False      # Any time a drawing is saved or loaded, 'quicksaving' is possible
        self.set_caption()              # Set the filename in the caption as blank
        self.clear_drawn_image()

    def save_drawing(self):
        if self.can_quicksave:
            self.save_drawing_2()       # Skip the filename entry and simply execute the 'saving' part based on the previous filename
        else:
            self.save_drawing_as()

    def save_drawing_as(self):
        self.pause_app()
        self.drawingname.active = True
        self.queued.append(self.save_drawing_2)

    def save_drawing_2(self):
        filename = self.drawingname.get()
        if filename[-4:] == ".txt":     # If the user puts ".txt" at the end, don't put ".txt.txt"
            filename = filename[:-4]
        try:                            # A try/except statement is used to catch unknown errors with the file, such as naming it "CON", which would otherwise pass these tests
            for char in "<>:\"/\\|?*":  # Forbidden characters for filenames
                if char in filename:
                    raise ValueError
            with open(filename + ".txt", "w") as file:
                print(self.answer_drawing, end="\n\n", file=file)
                for j in range(28):
                    print(" ".join(str(self.drawn_images[0][0][j][i])[7:-2] for i in range(28)), file=file)
                                                                    # [7:-2] extracts the number (e.g. tensor(-1) => -1)
            # Filename was valid and saving was successful
            self.can_quicksave = True
            self.set_caption(filename + ".txt")
        except:                         # Any other error with the filename is simply labelled "Invalid Filename"
            # Filename was invalid and saving was unsuccessful
            self.pause_app()
            self.msg.set_label("Invalid Filename")
            self.msg.active = True

    def open_drawing(self):
        self.pause_app()
        self.drawingname.active = True
        self.queued.append(self.open_drawing_2)

    def open_drawing_2(self):
        filename = self.drawingname.get()
        if filename[-4:] == ".txt":
            filename = filename[:-4]
        try:
            for char in "<>:\"/\\|?*":
                if char in filename:
                    raise ValueError
            with open(filename + ".txt") as file:
                for j, line in enumerate(file, -1): # Start counting from -1 so the first line has a different procedure
                    line = line.strip()
                    if j == -1:
                        # This is the first line of the file
                        if line == "None":
                            self.answer_drawing = None
                        else:
                            self.answer_drawing = int(line)
                    else:
                        # This is not the first line of the file
                        for i, value in enumerate(line.split(" ")):
                            self.draw(i, j, int(value))

            # Filename was valid and saving was successful
            self.can_quicksave = True
            self.set_caption(filename + ".txt")
        except:
            # Filename was invalid and saving was unsuccessful
            self.pause_app()
            self.msg.set_label("Invalid Filename")
            self.msg.active = True

    def draw_text(self, colour, pos, text, halign="LEFT", valign="TOP"):
        # Prepare variables for calculation
        x, y = pos
        halign = halign.upper()
        valign = valign.upper()
        render = self.font.render(text, 1, colour)  # Get the rendered surface
        w, h = render.get_size()                    # Get the size of this surface

        # Subtract the width and height from the x,y position to correctly align the text
        if halign == "RIGHT":
            x -= w
        elif halign == "CENTER" or halign == "CENTRE" or halign == "MIDDLE":
            x -= w//2
        if valign == "BOTTOM":
            y -= h
        elif valign == "CENTER" or valign == "CENTRE" or valign == "MIDDLE":
            y -= h//2

        self.screen.blit(render, (x, y))            # This function treats (x, y) as the top-left of where the result is rendered, which is why the previous subtractions are made

    def tick(self):
        self.inp.tick()
        if self.pause:
            for popup in self.popups:
                if popup.active:
                    popup.tick()
                    return

        while self.queued:                          # Run through queued events
            self.queued[0]()
            self.queued.pop(0)

        self.tick_menu()

        if self.mode == App.TRAINING:
            if self.rapid_training or self.inp.keys[pygame.K_SPACE]:    # Ensure 'rapid training + space' doesn't train twice as fast
                self.training_step()
        elif self.mode == App.VIDEO:
            self.tick_video()
        elif self.mode == App.DRAWING:
            self.tick_drawing()

    def tick_video(self):
        frame = self.capture.read()[1]              # The first returned variable indicates success or failure

        # The frame is landscape format, so cut it to make it square
        for row in frame:
            row = row[:frame.shape[1]]

        if self.grayscale():
            # Use grayscale frame to construct grayscale variable 'images'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Gray means 'gray BGR', doing this makes the format 'gray RGB'
            frame = cv2.resize(frame, dsize=(28, 28))       # Comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
            self.video_images = torch.empty(1, 1, 28, 28)   # Create new image, then loop through old image to populate new image
            for i in range(28):
                for j in range(28):
                    if self.video_mode == App.RAW:
                        self.video_images[0][0][i][j] = frame[i][j][0]/128 - 1                  # Convert from (0, 256) to (-1, 1)
                        self.render_images[App.VIDEO].set_at((j, i), (frame[i][j][0],)*3)       # Create tuple of grayscale value (e.g. 128 -> RGB(128, 128, 128))
                    elif self.video_mode == App.INVERT:
                        self.video_images[0][0][i][j] = 1 - frame[i][j][0]/128                  # Convert from (0, 256) to (1, -1)
                        self.render_images[App.VIDEO].set_at((j, i), (255-frame[i][j][0],)*3)   # Convert from (0, 256) to (256, 0)
                    elif self.video_mode == App.CUTOFF:
                        # Set to either black or white depending on cutoff
                        if frame[i][j][0] < self.cutoff.get():                                  # If less than the cutoff, set the pixel to black
                            self.video_images[0][0][i][j] = -1
                            self.render_images[App.VIDEO].set_at((j, i), (0, 0, 0))
                        else:                                                                   # Else, set the pixel to white
                            self.video_images[0][0][i][j] = 1
                            self.render_images[App.VIDEO].set_at((j, i), (255, 255, 255))
        else:
            # Use BGR frame to construct BGR variable 'images'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=(32, 32))       # Comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
            self.video_images = torch.empty(1, 3, 32, 32)
            for i in range(32):
                for j in range(32):
                    for c in range(3):
                        # Alternate video modes are not supported for non-grayscale datasets
                        self.video_images[0][c][i][j] = frame[i][j][c]/128 - 1

        self.trainer().guess_images(self.video_images, delay=5) # Only actually guess the image once in every 6 function calls

    def tick_drawing(self):
        # Compute which pixel of the image the mouse is over (<0 or >= 28 indicates not in the image)
        gx = (self.inp.mouse_x-self.image_left) * 28 // self.image_render_size
        gy = (self.inp.mouse_y-self.menu_height-self.padding) * 28 // self.image_render_size
        if 0 <= gx < 28 and 0 <= gy < 28:
            if self.inp.mouse[0]:                       # Left button held; draw
                # Set squares in the brush to either white (brush inside) or grey (brush edges)
                for i, j, c in self.brush:              # (i,j) is the displacement from the cursor, c is the colour to set that pixel to
                    if 0 <= gx+i < 28 and 0 <= gy+j < 28:
                        self.draw(gx+i, gy+j, max(c, self.drawn_images[0][0][gy+j][gx+i])) # max() is used so that white pixels don't get turned back into grey ones
            elif self.inp.mouse[2]:                     # Right button held; erase
                # Set a 5x5 square of pixels around the cursor to black (-1)
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if 0 <= gx+i < 28 and 0 <= gy+j < 28:
                            self.draw(gx+i, gy+j, -1)   # This function draws to both the image rendered and the image passed to the CNN

        self.trainer().guess_images(self.drawn_images)

    def draw(self, x, y, c):
        self.drawn_images[0][0][y][x] = c
        self.render_images[App.DRAWING].set_at((x, y), (min(255, (c+1)*128),)*3)    # min() is taken to prevent (1+1)*128 == 256, as 1 is set in drawings as 'white', but 256 is unacceptable

    def tick_menu(self):
        for menu_item in self.menu_items:
            menu_item.tick()

    def render(self):
        if self.pause:
            for popup in self.popups:
                if popup.active:
                    popup.render()
                    break
        else:
            self.screen.fill((255, 255, 128))           # Fill the background with light yellow
            self.render_image()                         # This method also renders the label under the image
            self.render_probabilities()
            self.render_statistics()
            self.render_menu()

    def render_image(self):
        # Create a black box around the image and its label
        self.screen.fill((0, 0, 0), rect=(
            self.image_left,
            self.menu_height + self.padding,
            self.image_render_size,
            self.image_render_size + self.font_size + 3*self.padding
        ))
        # Render image
        self.screen.blit(
            pygame.transform.scale(self.render_images[self.mode], (self.image_render_size, self.image_render_size)),
            (self.image_left, self.menu_height + self.padding)
        )
        # Render label centered under the image
        label = None
        if self.mode == App.TRAINING and self.trainer().started:
            label = self.CLASSES[self.dataset][self.trainer().labels[0]][0]
        elif self.mode == App.VIDEO and self.answer_video is not None:
            label = self.CLASSES[self.dataset][self.answer_video][0]
        elif self.mode == App.DRAWING and self.answer_drawing is not None:
            label = self.CLASSES[self.dataset][self.answer_drawing][0]
        if label is not None:
            self.draw_text(
                (255, 255, 255),
                (self.image_left + self.image_render_size//2, self.menu_height + self.image_render_size + 3*self.padding),
                label,
                halign="CENTER"
            )

    def render_probabilities(self):
        # Display each probability in descending order
        for i, probs in enumerate(zip(*torch.sort(self.trainer().probabilities[0], descending=True))):
            prob, probIndex = probs # fancy way of iterating such that 'i' goes from 0 to 9, 'probIndex' is the index of the class and 'prob' is the probability of that index
            top = self.menu_height + self.padding + i*(self.prob_height+self.padding)
            self.screen.fill((224, 224, 255), rect=(
                self.padding,
                top,
                self.prob_width,
                self.prob_height
            )) # Render light blue boxes; equal in size
            if self.trainer().started or self.mode != App.TRAINING:
                self.screen.fill((128, 128, 255), rect=(
                    self.padding,
                    top,
                    math.ceil(prob*self.prob_width),
                    self.prob_height)
                ) # Render dark blue boxes; the width depends on the probability value
                self.draw_text(
                    (0, 0, 0),
                    (self.prob_width, top + self.padding),
                    "{}: {: >5.2f}%".format(self.CLASSES[self.dataset][probIndex][0], 100*prob), # Right-align the % probability to 2 d.p.
                    halign="RIGHT",
                )

    def render_statistics(self):
        if self.trainer().started or self.mode != App.TRAINING:
            if self.mode == App.TRAINING:
                if self.trainer().correct_guess:
                    text_colour = (0, 160, 0)   # Green
                else:
                    text_colour = (224, 0, 0)   # Red
            elif self.mode == App.VIDEO:
                if self.answer_video is None:
                    text_colour = (0, 0, 0)     # Black
                elif self.answer_video == self.trainer().best_guess:
                    text_colour = (0, 160, 0)   # Green
                else:
                    text_colour = (224, 0, 0)   # Red
            elif self.mode == App.DRAWING:
                if self.answer_drawing is None:
                    text_colour = (0, 0, 0)     # Black
                elif self.answer_drawing == self.trainer().best_guess:
                    text_colour = (0, 160, 0)   # Green
                else:
                    text_colour = (224, 0, 0)   # Red
            self.draw_text(
                text_colour,
                (self.image_left, self.info_top),
                f"Best Guess: {self.CLASSES[self.dataset][self.trainer().best_guess][0]}"
            )
            enum_start = 1                      # Ensure that the text always renders in the correct order under the image
        else:
            enum_start = 0
        if self.mode == App.TRAINING and self.trainer().started:
            texts = [f"Loss: {self.trainer().loss:.5f}"]    # To 5 decimal places
        elif self.mode == App.VIDEO:
            texts = [f"Cutoff: {self.cutoff.get()}"]
        else:
            texts = []
        texts.append(f"Counter: {self.trainer().true_total}")
        texts.append(f"Acc: {self.trainer().correct}/{self.trainer().total} ({self.trainer().get_accuracy():>5.2f}%)")  # Right-aligned, 5 wide, to 2 decimal places
        for i, text in enumerate(texts, enum_start):
            self.draw_text(
                (0, 0, 0),
                (self.image_left, self.info_top + i*(self.font_size+self.padding)),
                text
            )

    def render_menu(self):
        # Draw the menu bounding box
        self.screen.fill((255, 255, 255), rect=(0, 0, self.width, self.menu_height))
        # Draw a line separating the menu and the application
        pygame.draw.line(
            self.screen,
            (128, 128, 128),
            (0, self.menu_height),
            (self.width, self.menu_height),
        )

        # Recursively render the entire menu structure, as each item renders its children
        for menu_item in self.menu_items:
            menu_item.render()

    def quit(self):
        self.inp.quit = True

    def destroy(self):
        pygame.quit()
