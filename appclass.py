import pygame
import cv2
import torch
import torchvision as tv

from trainer import * # Splitting up the program into different files makes it easier to find each piece
from inputclass import * # Python has an in-built 'input' function, so that name is unavailable
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

    def __init__(self, width):
        self.width = width

        self.load_datasets()
        self.init_pygame()
        self.init_app()

    def load_datasets(self):
        self.dataset = App.DIGIT # Dataset is changed using a drop-down menu

        # Torchvision conveniently allows the composition of all transforms to be applied to the training images into just one transform
        img_transforms = tv.transforms.Compose((
            tv.transforms.ToTensor(), # PyTorch's 'Tensors' are used due to their similarity to numpy arrays, which provide many benefits over multi-dimensional Python lists
                                      # The main benefit is the 'shape' attribute, which allows easy debugging of the network, as it is easy to identify the shape after convolutions
                                      # Another useful benefit is that printing numpy arrays or PyTorch tensors is much cleaner than lists of lists
            tv.transforms.Lambda(lambda x: 2*x - 1) # The values initially range from 0 to 1; after this transform, they range from -1 to 1
        ))
        # Define class arrays, dataloader arrays, and 'Trainer' object arrays, as tuples of length 3, one for each dataset
        self.CLASSES = (
            tuple(str(i) for i in range(10)), # ("0", "1", ..., "9")
            ("t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"),
            ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        )
        self.DATA_ITERATORS = (
            iter(torch.utils.data.DataLoader(
                tv.datasets.MNIST(root="./digit_data", train=True, transform=img_transforms, download=True), # train=True makes the iterator loop indefinitely
                batch_size=1,   # Train on one image at a time
                shuffle=True    # Randomise the order in which the images appear
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
        self.trainers = tuple(Trainer(self.DATA_ITERATORS[i], i!=2) for i in range(3)) # Create a tuple of 3 Trainers, one for each dataset
        # note that i!=2 determines whether the dataset is grayscale

    def trainer(self): # Get the trainer for the current dataset
        return self.trainers[self.dataset]

    def grayscale(self):
        return self.dataset != 2

    def init_pygame(self):
        pygame.init() # This is necessary to initialise fonts

        self.set_render_constants()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Real-Time CNN Image Classifier Application")
        self.font = pygame.font.SysFont("monospace", self.font_size)

    def set_render_constants(self):
        self.padding = self.width // 128 # All variables are defined in terms of width, including the height
        self.font_size = self.width // 64
        self.menu_height = self.font_size + 2*self.padding
        self.image_render_size = self.width // 4
        self.prob_width = self.width - self.image_render_size - 3*self.padding
        self.prob_height = self.font_size + 2*self.padding # Same as menu_height, but 2 variables exist to avoid confusion
        self.image_left = self.width - self.image_render_size - self.padding
        self.info_top = self.menu_height + self.image_render_size + self.font_size + 5*self.padding
        self.height = max(
            self.menu_height + 10*self.prob_height + 11*self.padding,
            self.info_top + 4*self.font_size + 4*self.padding
        ) # The max function is used to be safe, as it is unclear which column is lower, and this may vary with height

    def init_app(self):
        self.init_brush()
        self.init_capture()
        self.init_menu()

        self.inp = Input()

        self.mode = App.TRAINING
        self.rapid_training = False
        self.clear_image()

    def clear_image(self):
        self.drawn_images = torch.full((1, 1, 28, 28), -1)

    def init_brush(self):
        # Define the brush shape for drawing (xpos, ypos, colour) where colour==1 -> white and colour==0 -> gray
        self.brush = [(i, j, 1) for i in (0, 1) for j in (0, 1)]
        self.brush.extend((i, j, 0) for i in (0, 1) for j in (-1, 2))
        self.brush.extend((i, j, 0) for i in (-1, 2) for j in (0, 1))

    def init_capture(self):
        self.capture = cv2.VideoCapture(0) # 0 is the camera index, modifying it changes which camera is used for video capture
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1) # Set the resolution of the webcam as low as possible so that iterating through each frame is faster
        self.cutoff = 40 # (out of 256) any pixel darker will become black, any any pixel brighter will become white

    def init_menu(self):
        # Define menus which need a reference for later
        self.menu_cifar = Menu(self, "CIFAR-10", command=lambda:self.set_dataset(App.CIFAR))
        self.menu_drawing = Menu(self, "Drawing Mode", command=lambda:self.set_mode(App.DRAWING))
        self.cascmenus = (
            Menu(self, "Training", top=True, children=(
                Menu(self, "Execute Training Steps", children=(
                    Menu(self, "1 Step", command=self.multi_train_decorator(1)),
                ) + tuple(
                    Menu(self, f"{i} Steps", command=self.multi_train_decorator(i)) for i in (50, 100, 500, 1000, 5000, 10000, 50000)
                )),
                Menu(self, "Toggle Rapid Training", command=self.toggle_rapid_training),
                Menu(self, "Step Until Wrong", command=self.step_until_wrong)
            )),
            Menu(self, "Video", top=True, children=(
                Menu(self, "Not Implemented"),
            )),
            Menu(self, "Drawing", top=True, children=(
                Menu(self, "Clear Image", command=self.clear_image),
            ))
        )

        # Define the recursive GUI structure as a tuple of Menu items, some of which have children, which is a tuple of Menu items
        self.menu_items = [
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
                Menu(self, "Cutoff", children=(
                    Menu(self, "Increase", command=lambda:self.addto_cutoff(8)),
                    Menu(self, "Decrease", command=lambda:self.addto_cutoff(-8))
                ))
            )),
            Menu(self, "Edit", top=True, children=(
                Menu(self, "Reset Accuracy", command=lambda:self.trainer().reset_accuracy()),   # This must be inside a lambda so that self.trainer() is called each time
            )),                                                                                 # Otherwise, you could do 'command=self.trainer().reset_accuracy'
            self.cascmenus[0]
        ]

        # Place the top-level menu items horizontally across the top of the screen
        x = self.padding
        for menu_item in self.menu_items:
            menu_item.set_pos(x, 0) # y already is 0 but this is less confusing

            # Now that the positions of the top-level menu items are calculated, calculate the positions of the rest of the recursive structure
            if menu_item.has_children():
                menu_item.init_children()
            x += menu_item.w + 2*self.padding

        for menu_item in self.cascmenus[1:]:
            menu_item.set_pos(self.cascmenus[0].x, self.cascmenus[0].y)
            menu_item.init_children()

    # The following methods are necessary as they are passed as commands into the GUI Menu structure
    def set_mode(self, mode):
        if self.mode == mode: # If the app is already in this mode, do not bother with the rest of the calculations
            return
        self.mode = mode
        self.trainer().started = False
        if mode != App.TRAINING:
            self.rapid_training = False
        self.menu_cifar.disabled = (mode == App.DRAWING)
        self.menu_items[-1] = self.cascmenus[mode]

    def set_dataset(self, dataset):
        if self.dataset == dataset:
            return
        self.dataset = dataset
        self.rapid_training = False
        self.menu_drawing.disabled = (dataset == App.CIFAR)

    def addto_cutoff(self, num):
        self.cutoff += num
        if self.cutoff < 0:
            self.cutoff = 0
        elif self.cutoff > 256:
            self.cutoff = 256

    # An explanation of this is included at 'realtimecnn.blogspot.com' in the post 'Menus are harder than they look'
    def multi_train_decorator(self, num):
        def multi_train():
            for _ in range(num):
                self.trainer().training_step()
        return multi_train

    def toggle_rapid_training(self):
        self.rapid_training = not self.rapid_training

    def step_until_wrong(self):
        self.rapid_training = False # Cancel rapid training when this menu item is clicked
        for i in range(1000): # If the network achieves sufficiently high accuracy, or memorises the data, cap this at 1000 steps
            self.trainer().training_step() # Otherwise (e.g. using a while loop) an infinite loop may occur
            if not self.trainer().correct_guess:
                return
        # If the program makes it to here, nothing was guessed wrong. As an extension, I could get this to display a message praising the network

    def draw_text(self, colour, pos, text, halign="LEFT", valign="TOP"):
        # Prepare variables for calculation
        x, y = pos
        halign = halign.upper()
        valign = valign.upper()
        render = self.font.render(text, 1, colour) # Get the rendered surface
        w, h = render.get_size() # Get the size of this surface

        # Subtract the width and height from the x,y position to correctly align the text
        if halign == "RIGHT":
            x -= w
        elif halign == "CENTER" or halign == "CENTRE" or halign == "MIDDLE":
            x -= w//2
        if valign == "BOTTOM":
            y -= h
        elif valign == "CENTER" or valign == "CENTRE" or valign == "MIDDLE":
            y -= h//2

        self.screen.blit(render, (x, y)) # This function treats (x, y) as the top-left of where the result is rendered, which is why the previous subtractions are made

    def tick(self):
        self.inp.tick()
        self.tick_menu()

        if self.mode == App.TRAINING:
            if self.rapid_training or self.inp.keys[pygame.K_SPACE]: # Ensure 'rapid training + space' doesn't train twice as fast
                self.trainer().training_step()
        elif self.mode == App.VIDEO:
            self.tick_video()
        elif self.mode == App.DRAWING:
            self.tick_drawing()

    def tick_video(self):
        frame = self.capture.read()[1] # The first returned variable indicates success or failure

        # The frame is landscape format, so cut it to make it square
        for row in frame:
            row = row[:frame.shape[1]]

        if self.grayscale():
            # Use grayscale frame to construct grayscale variable 'images'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Gray means 'gray BGR', doing this makes the format 'gray RGB'
            self.video_raw = pygame.surfarray.make_surface(numpy.rot90(frame))
            frame = cv2.resize(frame, dsize=(28, 28)) # Comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
            self.video_images = torch.empty(1, 1, 28, 28)
            for i in range(28):
                for j in range(28):
                    # Set to either black or white depending on cutoff
                    if frame[i][j][0] < self.cutoff:
                        self.video_images[0][0][i][j] = -1
                    else:
                        self.video_images[0][0][i][j] = 1
                    #self.video_images[0][0][i][j] = frame[i][j][0]/128 - 1 # Normal colours, make customisable in menus as extension
                    #self.video_images[0][0][i][j] = 1 - frame[i][j][0]/128 # Inverted colours

        else:
            # Use BGR frame to construct BGR variable 'images'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_raw = pygame.surfarray.make_surface(numpy.rot90(frame))
            frame = cv2.resize(frame, dsize=(32, 32)) # Comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
            self.video_images = torch.empty(1, 3, 32, 32)
            for i in range(32):
                for j in range(32):
                    for c in range(3):
                        self.video_images[0][c][i][j] = frame[i][j][c]/128 - 1

        self.trainer().guess_images(self.video_images, delay=5) # only actually guess the image once in every 6 function calls

    def tick_drawing(self):
        # Compute which pixel of the image the mouse is over (<0 or >= 28 indicates not in the image)
        gx = (self.inp.mouse_x-self.image_left) * 28 // self.image_render_size
        gy = (self.inp.mouse_y-self.menu_height-self.padding) * 28 // self.image_render_size
        if 0 <= gx < 28 and 0 <= gy < 28:
            if self.inp.mouse[0]: # Left button held; draw
                # Set squares in the brush to either white (brush inside) or grey (brush edges)
                for i, j, c in self.brush: # (i,j) is the displacement from the cursor, c is the colour to set that pixel to
                    if 0 <= gx+i < 28 and 0 <= gy+j < 28:
                        self.drawn_images[0][0][gy+j][gx+i] = max(c, self.drawn_images[0][0][gy+j][gx+i]) # max() is used so that white pixels don't get turned back into grey ones
            elif self.inp.mouse[2]: # Right button held; erase
                # Set a 5x5 square of pixels around the cursor to black (-1)
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if 0 <= gx+i < 28 and 0 <= gy+j < 28:
                            self.drawn_images[0][0][gy+j][gx+i] = -1

        self.trainer().guess_images(self.drawn_images)

    def tick_menu(self):
        """
            Rather than running each menu's tick method, only the menus to the left of the cursor need to be checked
            This is because menus to the right of the cursor, even if they drop down, can never be hovered over or clicked
            This assumes that the menus are ordered from left to right in the list
        """

        x = self.padding
        for menu_item in self.menu_items:
            menu_item.tick()
            if x > self.inp.mouse_x:
                break
            x += menu_item.w

    def render(self):
        self.screen.fill((255, 255, 128)) # Fill the background with light yellow
        self.render_image() # This method also renders the label under the image
        self.render_probabilities()
        self.render_statistics()
        self.render_menu()

    def render_image(self):
        self.screen.fill((0, 0, 0), rect=(
            self.image_left,
            self.menu_height + self.padding,
            self.image_render_size,
            self.image_render_size + self.font_size + 3*self.padding
        )) # Create a black box around the image and its label
        render_image = None
        render_label = None
        if self.mode == App.TRAINING and self.trainer().started:
            render_image = self.trainer().images[0]
            render_label = self.trainer().labels[0]
        elif self.mode == App.VIDEO:
            render_image = self.video_images[0]
        elif self.mode == App.DRAWING:
            render_image = self.drawn_images[0]
        if render_image is not None:
            self.screen.blit(
                pygame.transform.scale( # Reformat the tensor/image so that pygame.surfarray.make_surface() accepts it, then enlarge it
                    pygame.surfarray.make_surface(tensor_to_image(render_image, self.grayscale())),
                    (self.image_render_size, self.image_render_size)
                ),
                (self.image_left, self.menu_height + self.padding) # Render it in the top right of the app
            )
        if render_label is not None:
            self.draw_text(
                (255, 255, 255),
                (self.image_left + self.image_render_size//2, self.menu_height + self.image_render_size + 3*self.padding),
                self.CLASSES[self.dataset][render_label],
                halign="CENTER"
            ) # Render label centered under the image

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
                    "{}: {: >5.2f}%".format(self.CLASSES[self.dataset][probIndex], 100*prob), # Right-align the % probability to 2 d.p.
                    halign="RIGHT",
                )

    def render_statistics(self):
        if self.trainer().started or self.mode != App.TRAINING:
            if self.mode != App.TRAINING:
                text_colour = (0, 0, 0) # Black
            elif self.trainer().correct_guess:
                text_colour = (0, 160, 0) # Green
            else:
                text_colour = (224, 0, 0) # Red
            self.draw_text(
                text_colour,
                (self.image_left, self.info_top),
                f"Best Guess: {self.CLASSES[self.dataset][self.trainer().best_guess]}"
            )
            enum_start = 1
        else:
            enum_start = 0
        if self.mode == App.TRAINING and self.trainer().started:
            texts = [f"Loss: {self.trainer().loss:.5f}"]
        elif self.mode == App.VIDEO:
            texts = [f"Cutoff: {self.cutoff}"]
        else:
            texts = []
        texts.append(f"Counter: {self.trainer().true_total}")
        texts.append(f"Acc: {self.trainer().correct}/{self.trainer().total} ({self.trainer().get_accuracy():>5.2f}%)")
        for i, text in enumerate(texts, enum_start):
            self.draw_text(
                (0, 0, 0),
                (self.image_left, self.info_top + i*(self.font_size+self.padding)),
                text
            )

    def render_menu(self):
        self.screen.fill((255, 255, 255), rect=(0, 0, self.width, self.menu_height))
        pygame.draw.line(
            self.screen,
            (128, 128, 128),
            (0, self.menu_height),
            (self.width, self.menu_height),
        )

        for menu_item in self.menu_items:
            menu_item.render()

    def destroy(self):
        pygame.quit()
