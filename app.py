import torch
import torchvision as tv
import pygame
import numpy
import math
import cv2

from trainer import * # splitting up the program into different files makes it easier to find each piece


"""
    Structure of the program:
        - The application is run using Pygame for two main reasons:
            1. Pygame supports the event handler/listener structure, which simplifies the handling of keyboard inputs.
            2. Rendering using Pygame is preferred to another GUI library such as Tkinter, as it is simpler to render. Using Pygame,
               the 'Surface' and 'Rect' classes and the 'font' module simplify rendering text, and bounding boxes for the text.
        - The application itself can be thought of as a class, but it is kept at the outer level as this is convenient for Python (as
          opposed to a more object-oriented language like Java, where classes are mandatory for each file).
        - The data is loaded using 'Torchvision', which provides many convenient functions that are commented below
        - The application defines a 'Trainer' object, which handles the training process, separate from the rendering in the application
        - Each Trainer object 'owns' a ImageClassifier object (i.e. has one as a property, as a 'self.image_classifier')
        - The ImageClassifier class is a Convolutional Neural Network, run using PyTorch to simplify many aspects of machine learning,
          such as backpropagation, and to make image handling intuitive by using numpy arrays (these are detailed below and in other files).
"""


# Create ordered tuples that convert from the indices used in the training data to string representations
DIGIT_CLASSES = tuple(str(i) for i in range(10)) # ("0", "1", ..., "9")
FASHION_CLASSES = ("t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")
CIFAR_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# ordered tuples are used here like dictionaries, where the key is the index (0 to 9 inclusive) provided as a label in the training data


# Initialise training datasets
img_transforms = tv.transforms.Compose(( # Torchvision conveniently allows the composition of all transforms to be applied to the training images into just one transform
    tv.transforms.ToTensor(), # PyTorch's 'Tensors' are used due to their similarity to numpy arrays, which provide many benefits over multi-dimensional Python lists
                              # The main benefit is the 'shape' attribute, which allows easy debugging of the network, as it is easy to identify the shape after convolutions
                              # Another useful benefit is that printing numpy arrays or PyTorch tensors is much cleaner than lists of lists
    tv.transforms.Lambda(lambda x: 2*x - 1) # the values initially range from 0 to 1; after this transform, they range from -1 to 1
))

digit_dataloader = torch.utils.data.DataLoader(
    tv.datasets.MNIST(root="./digit_data", train=True, transform=img_transforms, download=True), # train=True makes the iterator loop indefinitely
    batch_size=1,  # train on one image at a time
    shuffle=True # randomise the order in which the images appear
)

fashion_dataloader = torch.utils.data.DataLoader(
    tv.datasets.FashionMNIST(root="./fashion_data", train=True, transform=img_transforms, download=True),
    batch_size=1,
    shuffle=True
)

cifar_dataloader = torch.utils.data.DataLoader(
    tv.datasets.CIFAR10(root="./cifar_data", train=True, transform=img_transforms, download=True),
    batch_size=1,
    shuffle=True
)

# Redefine these variables (and others) to test with different datasets (more info provided in user manual)
classes = CIFAR_CLASSES
training_data_iter = iter(cifar_dataloader) # create an iterator to iterate through the data


# Define render constants (see user manual for how to change dimensions)
WIDTH = 1024
IMAGE_RENDER_SIZE = WIDTH // 4 # all variables are defined in terms of width, including the height
PADDING = WIDTH // 128
FONT_SIZE = WIDTH // 64
BOX_HEIGHT = 2*PADDING + FONT_SIZE
BOX_WIDTH = WIDTH - 3*PADDING - IMAGE_RENDER_SIZE
INSTRUCTIONS_TOP = max(10*BOX_HEIGHT, 11*PADDING, IMAGE_RENDER_SIZE + 5*FONT_SIZE + 9*PADDING) # the max function is used to be safe, as it is unclear which column of the app is lower, and this may vary with height
HEIGHT = INSTRUCTIONS_TOP + 8*FONT_SIZE + 10*PADDING

# Initialise pygame fonts and display
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CNN Application")
DEFAULT_FONT = pygame.font.SysFont("monospace", FONT_SIZE)

# Define useful functions for the display
def tensor_to_image(tensor):
    """
        The parameter 'tensor' is a tensor representing an image.
        This is transposed, normalised and reformatted into the output 'img', which is a numpy array.
        The shape of the input tensor is [colours (of which there are 3), rows, columns].
        That means that the tensor consists of three arrays, one for each colour, each of which contains rows, each of which contains columns.
        The order of the dimensions is redefined (this is called 'transposing') to be [columns, rows, colours], for use in pygame.
        As well as this, 1 is added to each value from -1 to 1, making the range 0 to 2, then multiplying by 128 gives the correct colour range 0 to 256.
        As a failsafe, the colour is restrained so that it cannot be greater than 255.
    """

    img = numpy.empty((tensor.shape[2], tensor.shape[1], 3), "uint8") # prepare the numpy array, setting its type to integers
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(3):
                if tensor.shape[0] == 3:
                    # If the original tensor is rgb (3 colours), copy those colours into the new array
                    img[y][x][c] = min(255, (tensor[c][x][y] + 1) * 128)
                elif tensor.shape[0] == 1:
                    # Else if the original tensor is grayscale (1 colour), copy that colour into the new array 3 times for r, g, and b
                    img[y][x][c] = min(255, (tensor[0][x][y] + 1) * 128)
                else:
                    # Raise an error for unsupported tensor shapes
                    raise ValueError("Currently supported only for tensors with 1 channel (grayscale) or 3 channels (rgb)")

    return img


def draw_text(surface, colour, pos, text, font=DEFAULT_FONT, halign="LEFT", valign="TOP"):
    # Prepare variables for calculation
    x, y = pos
    halign = halign.upper()
    valign = valign.upper()
    render = font.render(text, 1, colour) # render the text
    w, h = render.get_size() # get the size of the rendered text

    # subtract the width and height from the x,y position to correctly align the text
    if halign == "RIGHT":
        x -= w
    elif halign == "CENTER" or halign == "CENTRE" or halign == "MIDDLE":
        x -= w//2
    if valign == "BOTTOM":
        y -= h
    elif valign == "CENTER" or valign == "CENTRE" or valign == "MIDDLE":
        y -= h//2

    surface.blit(render, (x, y)) # this function treats (x, y) as the top-left of where the result is rendered, which is why the previous subtractions are made


# Initialise application variables
running = True
trainer = Trainer(training_data_iter) # create an instance of the 'Trainer' class for use in the application
started = False # when the first training step begins, this becomes True

vidcap = cv2.VideoCapture(0) # 0 is the camera index, modifying it changes which camera is used for video capture
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1)

# Keys
key_space = False # whether the space key is held (not just pressed)
key_shift_toggle = False

# Application loop
while True:
    # Reset the 'pressed' keys
    key_s = False

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            if event.key == pygame.K_s:
                key_s = True
            elif event.key == pygame.K_SPACE:
                key_space = True
            elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                key_shift_toggle = not key_shift_toggle
            elif event.key == pygame.K_h:
                # Execute a hundred training steps
                for i in range(100):
                    trainer.training_step()
                started = True
            elif event.key == pygame.K_t:
                # Execute a thousand training steps
                for i in range(1000):
                    trainer.training_step()
                started = True
            elif event.key == pygame.K_e:
                # Execute ten thousand training steps (1 epoch for each dataset)
                for i in range(10000):
                    trainer.training_step()
                started = True
            elif event.key == pygame.K_w:
                while True:
                    trainer.training_step()
                    if not trainer.correct_guess:
                        break
                started = True
            elif event.key == pygame.K_r:
                trainer.reset_accuracy()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                key_space = False

    # Exit the program by terminating the loop if any event changes running to False
    if not running:
        break

    if key_shift_toggle:
        """# NEW STUFF
        frame = vidcap.read()[1] # the first returned variable indicates success or failure
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # gray means 'gray BGR', doing this makes the format 'gray RGB'
        #frame = cv2.resize(frame, dsize=(32, 32)) # comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
        images = torch.empty(1, 1, 28, 28)
        for i in range(28):
            for j in range(28):
                images[0][0][i][j] = frame[i][j][0]/128 - 1
                #images[0][0][i][j] = 1 - frame[i][j][0]/128 # inverted colours"""

        # NEW STUFF
        frame = vidcap.read()[1] # the first returned variable indicates success or failure
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, dsize=(32, 32)) # comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
        images = torch.empty(1, 3, 32, 32)
        for c in range(3):
            for i in range(32):
                for j in range(32):
                    images[0][c][i][j] = frame[i][j][c]/128 - 1

        trainer.guess_images(images)

    # Execute a training step each frame the space key is held
    elif key_s or key_space: # this is to prevent multiple keys being held at the same time executing multiple steps and skipping rendering
        trainer.training_step()
        started = True

    # Set colour for text rendering to green (correct) or red (incorrect)
    if started:
        if trainer.correct_guess:
            text_colour = (0, 160, 0)
        else:
            text_colour = (224, 0, 0)


    # Render the training results to the display
    screen.fill((255, 255, 128))

    # Reformat the tensor so that pygame.surfarray.make_surface() accepts it, then increase the size and render it in the top-right
    screen.fill((0, 0, 0), rect=(WIDTH-IMAGE_RENDER_SIZE-PADDING, PADDING, IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + FONT_SIZE + 3*PADDING))
    if started:
        if key_shift_toggle:
            render_image = images[0]
        else:
            render_image = trainer.images[0]
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(tensor_to_image(render_image)), (IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE)), (WIDTH-IMAGE_RENDER_SIZE-PADDING, PADDING))
        draw_text(screen, (255, 255, 255), (WIDTH - PADDING - IMAGE_RENDER_SIZE//2, IMAGE_RENDER_SIZE + 3*PADDING), classes[trainer.labels[0]], halign="CENTER") # render the correct label under the image

    # Display each probability in descending order
    if started:
        for i, probs in enumerate(zip(*torch.sort(trainer.probabilities[0], descending=True))):
            prob, probIndex = probs # fancy way of iterating such that 'i' goes from 0 to 9, 'probIndex' is the index of the class and 'prob' is the probability of that index
            screen.fill((224, 224, 255), rect=(PADDING, PADDING + i*(BOX_HEIGHT+PADDING), BOX_WIDTH, BOX_HEIGHT))
            screen.fill((128, 128, 255), rect=(PADDING, PADDING + i*(BOX_HEIGHT+PADDING), math.ceil(prob*BOX_WIDTH), BOX_HEIGHT))
            draw_text(screen, (0, 0, 0), (BOX_WIDTH, PADDING + i*(BOX_HEIGHT+PADDING) + BOX_HEIGHT//2), "{}: {: >5.2f}%".format(classes[probIndex], 100*prob), halign="RIGHT", valign="MIDDLE")
    else:
        # Render only the light blue rectangle if the program has not started yet
        for i in range(10):
            screen.fill((224, 224, 255), rect=(PADDING, PADDING + i*(BOX_HEIGHT+PADDING), BOX_WIDTH, BOX_HEIGHT))

    # Render training statistics
    if started:
        draw_text(screen, text_colour, (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + FONT_SIZE + 5*PADDING), "Best Guess: " + classes[trainer.best_guess])
        draw_text(screen, (0, 0, 0), (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + 2*FONT_SIZE + 6*PADDING), "Loss: {:.5f}".format(trainer.loss))
        draw_text(screen, (0, 0, 0), (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + 3*FONT_SIZE + 7*PADDING), "Counter: {}".format(trainer.true_total))
        draw_text(screen, (0, 0, 0), (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + 4*FONT_SIZE + 8*PADDING), "Acc: {: >5.2f}% ({}/{})".format(trainer.get_accuracy(), trainer.correct, trainer.total))

    # Render instructions
    # Note that the exact coordinate of each object on the screen is known, and this is the advantage of using Pygame as a GUI
    # Instead of the 'web design-style' where objects are given margins and padding, and are placed seemingly where it wants to be placed, I hardcode each object's position
    # I believe this makes the application look structured and nice, while also being intuitive
    screen.fill((160, 255, 160), rect=(PADDING, INSTRUCTIONS_TOP, WIDTH - 2*PADDING, 8*FONT_SIZE + 9*PADDING))
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + PADDING), "Press 'S' for a single training step")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + FONT_SIZE + 2*PADDING), "Hold 'Space' or press 'Shift' to toggle rapid training")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 2*FONT_SIZE + 3*PADDING), "Press 'H' for 100 steps")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 3*FONT_SIZE + 4*PADDING), "Press 'T' for 1,000 steps")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 4*FONT_SIZE + 5*PADDING), "Press 'E' for an epoch (10,000 steps)")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 5*FONT_SIZE + 6*PADDING), "Press 'W' to step until a wrong answer")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 6*FONT_SIZE + 7*PADDING), "Press 'R' to reset accuracy calculations")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 7*FONT_SIZE + 8*PADDING), "Press 'Esc' to Quit")
    
    pygame.display.update()

pygame.quit()
