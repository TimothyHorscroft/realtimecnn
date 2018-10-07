import torch
import numpy


def point_in_rect(point, rect): # Check if a point is inside a bounding box
    """
        NOTE: The rectangle uses WIDTH and HEIGHT, not (x1, y1, x2, y2) coordinates
    """

    x1, y1 = point
    x2, y2, w, h = rect
    return x2 <= x1 < x2 + w and y2 <= y1 < y2 + h


def tensor_to_image(tensor, grayscale):
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
                if grayscale:
                    # Copy that colour into the new array 3 times for r, g and b
                    img[y][x][c] = min(255, (tensor[0][x][y] + 1) * 128) # capping this at 255 is necessary as drawing white uses 1 which turns into 256
                else:
                    # Copy those colours into the new array
                    img[y][x][c] = min(255, (tensor[c][x][y] + 1) * 128)

    return img
