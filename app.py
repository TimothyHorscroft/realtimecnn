import pygame
import torch
import torchvision as tv
import cv2
import numpy
import math

from trainer import * # splitting up the program into different files makes it easier to find each piece


"""
    Structure of the program:
        - The MVP uses Pygame, but the final project will use Tkinter
        - The application itself can be thought of as a class, but it is kept at the outer level as this is convenient for Python (as
          opposed to a more object-oriented language like Java, where CLASSES are mandatory for each file). When the application is
          converted to Tkinter, it will likely assume this more object-oriented form.
        - The data is loaded using 'Torchvision', which provides many convenient functions that are commented below
        - The application defines a 'Trainer' object, which handles the training process, separate from the rendering in the application
        - Each Trainer object 'owns' a ImageClassifier object (i.e. has one as a property, as a 'self.image_classifier')
        - The ImageClassifier class is a Convolutional Neural Network, run using PyTorch to simplify many aspects of machine learning,
          such as backpropagation, and to make image handling intuitive by using numpy arrays (these are detailed below and in other files).
"""


# Redefining 'DATASET' updates all other aspects of code
DATASET = 0 # 0: MNIST Digits, 1: MNIST Fashion Items, 2: CIFAR Real-World Objects
GRAYSCALE = (DATASET != 2) # Grayscale unless dataset 2 (CIFAR) is chosen


# Define the brush shape for drawing (xpos, ypos, colour) where colour==1 -> white and colour==0 -> gray
brush = [(i, j, 1) for i in (0, 1) for j in (0, 1)]
brush.extend((i, j, 0) for i in (0, 1) for j in (-1, 2))
brush.extend((i, j, 0) for i in (-1, 2) for j in (0, 1))

# Define render constants (see user manual for how to change dimensions)
WIDTH = 1024
IMAGE_RENDER_SIZE = WIDTH // 4 # all variables are defined in terms of width, including the height
PADDING = WIDTH // 128
FONT_SIZE = WIDTH // 64
BOX_HEIGHT = 2*PADDING + FONT_SIZE
BOX_WIDTH = WIDTH - 3*PADDING - IMAGE_RENDER_SIZE
INSTRUCTIONS_TOP = max(10*BOX_HEIGHT, 11*PADDING, IMAGE_RENDER_SIZE + 5*FONT_SIZE + 9*PADDING) # the max function is used to be safe, as it is unclear which column of the app is lower, and this may vary with height
HEIGHT = INSTRUCTIONS_TOP + 10*FONT_SIZE + 12*PADDING

# Initialise pygame fonts and display
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time CNN Image Classifier Application")
DEFAULT_FONT = pygame.font.SysFont("monospace", FONT_SIZE)

# Initialise training datasets
img_transforms = tv.transforms.Compose(( # Torchvision conveniently allows the composition of all transforms to be applied to the training images into just one transform
    tv.transforms.ToTensor(), # PyTorch's 'Tensors' are used due to their similarity to numpy arrays, which provide many benefits over multi-dimensional Python lists
                              # The main benefit is the 'shape' attribute, which allows easy debugging of the network, as it is easy to identify the shape after convolutions
                              # Another useful benefit is that printing numpy arrays or PyTorch tensors is much cleaner than lists of lists
    tv.transforms.Lambda(lambda x: 2*x - 1) # the values initially range from 0 to 1; after this transform, they range from -1 to 1
))
if DATASET == 0:
    # CLASSES is an ordered tuple which converts from the indices used in the training data (0 to 9 inclusive) to class labels for the training data
    CLASSES = tuple(str(i) for i in range(10)) # ("0", "1", ..., "9")
    # ############### COMMENT ABOUT DATALOADER GOES HERE DONT FORGET AGAIN @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    dataloader = torch.utils.data.DataLoader(
        tv.datasets.MNIST(root="./digit_data", train=True, transform=img_transforms, download=True), # train=True makes the iterator loop indefinitely
        batch_size=1,   # train on one image at a time
        shuffle=True    # randomise the order in which the images appear
    )
elif DATASET == 1:
    CLASSES = ("t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")
    dataloader = torch.utils.data.DataLoader(
        tv.datasets.FashionMNIST(root="./fashion_data", train=True, transform=img_transforms, download=True),
        batch_size=1,
        shuffle=True
    )
elif DATASET == 2:
    CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    dataloader = torch.utils.data.DataLoader(
        tv.datasets.CIFAR10(root="./cifar_data", train=True, transform=img_transforms, download=True),
        batch_size=1,
        shuffle=True
    )
training_data_iter = iter(dataloader)


# Define useful functions for the display
def tensor_to_image(tensor):
    """
        The parameter 'tensor' is a tensor representing an image.
        This is transposed, normalised and reformatted into the output 'img', which is a numpy array.
        The shape of the input tensor is [colours, rows, columns].
        That means that the tensor consists of three arrays, one for each colour, each of which contains rows, each of which contains columns.
        The order of the dimensions is redefined (this is called 'transposing') to be [columns, rows, colours], for use in Pygame.
        As well as this, 1 is added to each value from -1 to 1, making the range 0 to 2, then multiplying by 128 gives the correct colour range 0 to 256.
    """

    img = numpy.empty((tensor.shape[2], tensor.shape[1], 3), "uint8") # prepare the numpy array, setting its type to integers
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(3):
                if GRAYSCALE:
                    # Copy that colour into the new array 3 times for r, g and b
                    img[y][x][c] = min(255, (tensor[0][x][y] + 1) * 128) # capping this at 255 is necessary as drawing white uses 1 which turns into 256
                else:
                    # Copy those colours into the new array
                    img[y][x][c] = min(255, (tensor[c][x][y] + 1) * 128)

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
started = False # when the first training step begins, this becomes True
cutoff = 40 # (out of 256) any pixel darker will become black, any any pixel brighter will become white

# Keys
key_space = False # whether the space key is held (not just pressed)
key_shift_toggle = 0 # toggles between 0: training mode, 1: video mode, 2: drawing mode
mouse_left = False
mouse_right = False

# Initialise application objects
trainer = Trainer(training_data_iter, GRAYSCALE) # create an instance of the 'Trainer' class for use in the application
vidcap = cv2.VideoCapture(0) # 0 is the camera index, modifying it changes which camera is used for video capture
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1) # set the resolution of the webcam as low as possible so that iterating through each frame is faster
drawn_images = torch.full((1, 1, 28, 28), -1)


# Application loop
while True:
    # Initialise/reset the 'pressed' keys
    key_s = False

    # Get mouse pos
    mx, my = pygame.mouse.get_pos()

    # Handle events (this is messy and long, but will not be in the final version, as Tkinter menus will be used)
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
                key_shift_toggle += 1
                key_shift_toggle %= 3
            elif key_shift_toggle == 0:
                if event.key == pygame.K_h:
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
                    # Executing training steps until the network guesses wrong
                    while True:
                        trainer.training_step()
                        if not trainer.correct_guess:
                            break # this is essentially a 'do-while' loop
                    started = True
            elif event.key == pygame.K_r:
                trainer.reset_accuracy()
            elif event.key == pygame.K_z:
                cutoff -= 8
            elif event.key == pygame.K_x:
                cutoff += 8
            elif event.key == pygame.K_c:
                # Clear the drawn image
                drawn_images = torch.full((1, 1, 28, 28), -1)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                key_space = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_left = True
            elif event.button == 3:
                mouse_right = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_left = False
            elif event.button == 3:
                mouse_right = False
    # Exit the program by terminating the loop if any event changes running to False
    if not running:
        break           # this is preferred over 'while running:' as this quits immediately after a pygame.QUIT event, without further calculation

    if key_shift_toggle == 0 and key_s or key_space: # this is to ensure that only one training_step occurs per frame
        # TRAINING MODE

        trainer.training_step()
        started = True
    elif key_shift_toggle == 1:
        # VIDEO MODE

        frame = vidcap.read()[1] # the first returned variable indicates success or failure

        # The image is landscape format, so cut it to make it square
        for row in frame:
            row = row[:frame.shape[1]]

        if GRAYSCALE:
            # Use grayscale frame to construct grayscale variable 'images'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # gray means 'gray BGR', doing this makes the format 'gray RGB'
            image = pygame.surfarray.make_surface(numpy.rot90(frame))
            frame = cv2.resize(frame, dsize=(28, 28)) # comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
            images = torch.empty(1, 1, 28, 28)
            for i in range(28):
                for j in range(28):
                    # Set to either black or white depending on cutoff
                    if frame[i][j][0] < cutoff:
                        images[0][0][i][j] = -1
                    else:
                        images[0][0][i][j] = 1
                    #images[0][0][i][j] = frame[i][j][0]/128 - 1 # normal colours
                    #images[0][0][i][j] = 1 - frame[i][j][0]/128 # inverted colours

        else:
            # Use BGR frame to construct BGR variable 'images'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = pygame.surfarray.make_surface(numpy.rot90(frame))
            frame = cv2.resize(frame, dsize=(32, 32)) # comment for clear image, bad FOV, uncomment for bad pixel image, good FOV
            images = torch.empty(1, 3, 32, 32)
            for i in range(32):
                for j in range(32):
                    for c in range(3):
                        images[0][c][i][j] = frame[i][j][c]/128 - 1

        trainer.guess_images(images, delay=5) # only actually guess the image once in every six function calls

    elif key_shift_toggle == 2 and GRAYSCALE:
        # DRAWING MODE (only works in grayscale)

        LEFT = WIDTH - IMAGE_RENDER_SIZE - PADDING
        TOP = PADDING
        # Compute which pixel of the image the mouse is over (<0 or >= 28 indicates not in the image)
        gx = (mx-LEFT) * drawn_images.shape[2] // IMAGE_RENDER_SIZE
        gy = (my-TOP) * drawn_images.shape[3] // IMAGE_RENDER_SIZE
        if 0 <= gx < 28 and 0 <= gy < 28:
            if mouse_left:
                # set squares in the brush
                for i, j, c in brush: # (i,j) is the displacement from the cursor, c is the colour to set that pixel to
                    if 0 <= gx+i < 28 and 0 <= gy+j < 28:
                        drawn_images[0][0][gy+j][gx+i] = max(c, drawn_images[0][0][gy+j][gx+i])
            elif mouse_right:
                # set 5x5 square of pixels around cursor to black (-1)
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if 0 <= gx+i < 28 and 0 <= gy+j < 28:
                            drawn_images[0][0][gy+j][gx+i] = -1

        trainer.guess_images(drawn_images)

    # Set colour for text rendering to green (correct) or red (incorrect)
    if trainer.correct_guess:
        text_colour = (0, 160, 0)
    else:
        text_colour = (224, 0, 0)


    # Render the training results to the display
    screen.fill((255, 255, 128))

    screen.fill((0, 0, 0), rect=(WIDTH-IMAGE_RENDER_SIZE-PADDING, PADDING, IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + FONT_SIZE + 3*PADDING))
    render_image = None
    render_label = None
    if key_shift_toggle == 0 and started:
        render_image = trainer.images[0]
        render_label = trainer.labels[0]
    elif key_shift_toggle == 1:
        render_image = images[0]
    elif key_shift_toggle == 2 and GRAYSCALE:
        render_image = drawn_images[0]
    if render_image is not None:
        # Reformat the tensor so that pygame.surfarray.make_surface() accepts it, then increase the size and render it in the top-right
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(tensor_to_image(render_image)), (IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE)), (WIDTH-IMAGE_RENDER_SIZE-PADDING, PADDING))
    if render_label is not None:
        # Render the correct label under the image
        draw_text(screen, (255, 255, 255), (WIDTH - PADDING - IMAGE_RENDER_SIZE//2, IMAGE_RENDER_SIZE + 3*PADDING), CLASSES[render_label], halign="CENTER")

    # Display each probability in descending order
    if started:
        for i, probs in enumerate(zip(*torch.sort(trainer.probabilities[0], descending=True))):
            prob, probIndex = probs # fancy way of iterating such that 'i' goes from 0 to 9, 'probIndex' is the index of the class and 'prob' is the probability of that index
            screen.fill((224, 224, 255), rect=(PADDING, PADDING + i*(BOX_HEIGHT+PADDING), BOX_WIDTH, BOX_HEIGHT))
            screen.fill((128, 128, 255), rect=(PADDING, PADDING + i*(BOX_HEIGHT+PADDING), math.ceil(prob*BOX_WIDTH), BOX_HEIGHT))
            draw_text(screen, (0, 0, 0), (BOX_WIDTH, PADDING + i*(BOX_HEIGHT+PADDING) + BOX_HEIGHT//2), "{}: {: >5.2f}%".format(CLASSES[probIndex], 100*prob), halign="RIGHT", valign="MIDDLE")
    else:
        # Render only the light blue rectangle if the program has not started yet
        for i in range(10):
            screen.fill((224, 224, 255), rect=(PADDING, PADDING + i*(BOX_HEIGHT+PADDING), BOX_WIDTH, BOX_HEIGHT))

    # Render training statistics
    if started:
        draw_text(screen, text_colour, (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + FONT_SIZE + 5*PADDING), "Best Guess: " + CLASSES[trainer.best_guess])
        if trainer.loss >= 0:
            draw_text(screen, (0, 0, 0), (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + 2*FONT_SIZE + 6*PADDING), "Loss: {:.5f}".format(trainer.loss))
    draw_text(screen, (0, 0, 0), (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + 3*FONT_SIZE + 7*PADDING), "Counter: {}".format(trainer.true_total))
    draw_text(screen, (0, 0, 0), (WIDTH - PADDING - IMAGE_RENDER_SIZE, IMAGE_RENDER_SIZE + 4*FONT_SIZE + 8*PADDING), "Acc: {: >5.2f}% ({}/{})".format(trainer.get_accuracy(), trainer.correct, trainer.total))

    # Render instructions
    # Doing this in Pygame is easy as the exact coordinates of each thing on the screen are known
    # This will be one annoying consequence of moving to Tkinter, though I will still attempt to keep the program structured
    screen.fill((160, 255, 160), rect=(PADDING, INSTRUCTIONS_TOP, WIDTH - 2*PADDING, 10*FONT_SIZE + 11*PADDING))
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + PADDING), "Press 'S' for a single training step")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + FONT_SIZE + 2*PADDING), "Hold 'Space' for rapid training")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 2*FONT_SIZE + 3*PADDING), "Press 'H' for 100 steps")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 3*FONT_SIZE + 4*PADDING), "Press 'T' for 1,000 steps")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 4*FONT_SIZE + 5*PADDING), "Press 'E' for an epoch (10,000 steps)")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 5*FONT_SIZE + 6*PADDING), "Press 'W' to step until a wrong answer")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 6*FONT_SIZE + 7*PADDING), "Press 'R' to reset accuracy calculations")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 7*FONT_SIZE + 8*PADDING), "Press 'Shift' to change mode")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 8*FONT_SIZE + 9*PADDING), "Use 'Z' and 'X' to change the cutoff in Video Mode")
    draw_text(screen, (0, 0, 0), (2*PADDING, INSTRUCTIONS_TOP + 9*FONT_SIZE + 10*PADDING), "Press 'Esc' to Quit")

    # Render which mode the user is in
    if key_shift_toggle == 0:
        render_text = "Training Mode"
    elif key_shift_toggle == 1:
        render_text = "Video Mode"
    else:
        render_text = "Drawing Mode"
    draw_text(screen, (0, 0, 0), (WIDTH - 2*PADDING, HEIGHT - 2*PADDING), render_text, halign="RIGHT", valign="BOTTOM")

    # If in video mode, display the video capture (and show the cutoff) to the top-left of the screen
    if key_shift_toggle == 1:
        screen.blit(image, (WIDTH - image.get_width() - 2*PADDING, INSTRUCTIONS_TOP + PADDING))
        draw_text(screen, (0, 0, 0), (WIDTH - 2*PADDING, INSTRUCTIONS_TOP + image.get_height() + 2*PADDING), f"Cutoff: {cutoff}", halign="RIGHT")
    
    pygame.display.update()

pygame.quit()
