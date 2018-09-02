import torch


# The Convolutional Neural Networks (CNNs)
# The Grayscale classifier works with the MNIST digits and fashion datasets, as their images are 1x28x28 (i.e. 1 colour channel)
class GrayscaleImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # These layers use adjustable filters to shrink the image, resulting in faster and more effective training that a Multi-Layer Perceptron (MLP)
        self.conv_layer1 = torch.nn.Conv2d(1, 3, 5) # Take 1 channel as input and use discrete 2D convolution to output 3 channels, using a 5x5 kernel
        self.conv_layer2 = torch.nn.Conv2d(3, 6, 5) # Intended for use after the first layer, where there are 3 channels. After this, there are 6.

        self.pool = torch.nn.MaxPool2d(2, stride=2) # Use a 2D max pooling algorithm, with a 2x2 kernel and a stride of 2

        self.fc1 = torch.nn.Linear(6 * 10 * 10, 120) # See comments in ImageClassifier.forward for how the first number is obtained
        self.fc2 = torch.nn.Linear(120, 30) # These are 'fully-connected layers', similar to MLPs
        self.fc3 = torch.nn.Linear(30, 10) # While passing through the fully-connected layers, the shape goes from 1x600 -> 1x120 -> 1x30 -> 1x10 (output)

    def forward(self, x):
        """
            When defining a subclass of torch.nn.Module, a 'forward' method must be defined.
            This method defines the CNN's structure, i.e. the path that inputs take as they move through the CNN and become outputs
            The advantage of using the PyTorch library is that the relatively complicated aspects of neural networks (i.e. backpropagation,
            stochastic gradient descent, etc.) are dramatically simplified by the existence of many useful functions that come with the
            module, and so more focus can be put on the structure of the CNN, and the functions specific to it (i.e. convolution, max
            pooling and ReLU).
        """

        x = self.conv_layer1(x) # After this layer, the image is resized from [1, 28, 28] to [3, 24, 24]
        x = torch.nn.functional.relu(x) # Set all negative values to 0, i.e. x = max(0, x)
        
        x = self.conv_layer2(x) # The image is resized from [3, 24, 24] to [6, 20, 20]
        x = torch.nn.functional.relu(x)
        
        x = self.pool(x) # Because the kernel size and stride are both 2, the image size is halved in 2 dimensions to [6, 10, 10]

        x = x.view(1, 6 * 10 * 10) # Reshape the image from [6, 10, 10] to [1, 6*10*10] so it can be passed through the fully-connected layers

        x = self.fc1(x) # The image is resized from [1, 600] to [1, 120]
        x = torch.nn.functional.relu(x)
        
        x = self.fc2(x) # The image is resized from [1, 120] to [1, 30]
        x = torch.nn.functional.relu(x)
        
        x = self.fc3(x) # The image is resized from [1, 30] to its final output form [1, 10] (as there are 10 classes in each dataset)

        return x


# The RGB classifier works with the CIFAR dataset, as its images are 3x32x32 (i.e. 3 colour channels, red, green and blue)
class RGBImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer1 = torch.nn.Conv2d(3, 8, 5) # the shape of the layer during the forward pass is described in the forward function
        self.conv_layer2 = torch.nn.Conv2d(8, 15, 5)
        
        self.pool = torch.nn.MaxPool2d(2, stride=2)

        self.fc1 = torch.nn.Linear(15 * 5 * 5, 80)
        self.fc2 = torch.nn.Linear(80, 30)
        self.fc3 = torch.nn.Linear(30, 10) # While passing through the fully-connected layers, the shape goes from 1x375 -> 1x80 -> 1x30 -> 1x10 (output)

    def forward(self, x):
        """
            When defining a subclass of torch.nn.Module, a 'forward' method must be defined.
            This method defines the CNN's structure, i.e. the path that inputs take as they move through the CNN and become outputs
            The advantage of using the PyTorch library is that the relatively complicated aspects of neural networks (i.e. backpropagation,
            stochastic gradient descent, etc.) are dramatically simplified by the existance of many useful functions that come with the
            module, and so more focus can be put on the structure of the CNN, and the functions specific to it (i.e. convolution,
            max pooling and RELU).
        """

        x = self.conv_layer1(x) # After this layer, the image is resized from [3, 32, 32] to [5, 28, 28]
        x = torch.nn.functional.relu(x) # Set all negative values to 0, i.e. x = max(0, x)

        x = self.pool(x) # The image is resized from [5, 28, 28] to [5, 14, 14]
        
        x = self.conv_layer2(x) # The image is resized from [5, 14, 14] to [15, 10, 10]
        x = torch.nn.functional.relu(x)
        
        x = self.pool(x) # The image is resized from [15, 10, 10] to [15, 5, 5]

        x = x.view(1, 15 * 5 * 5) # Reshape the image from [15, 5, 5] to [1, 15*5*5] so it can be passed through the fully-connected layers

        x = self.fc1(x) # The image is resized from [1, 375] to [1, 80]
        x = torch.nn.functional.relu(x)
        
        x = self.fc2(x) # The image is resized from [1, 80] to [1, 30]
        x = torch.nn.functional.relu(x)
        
        x = self.fc3(x) # The image is resized from [1, 30] to its final output form [1, 10] (as there are 10 classes in each dataset)

        return x


# If any custom CNNs are to be created by the user of this application, by following the above two networks as templates, a third network can be created.
