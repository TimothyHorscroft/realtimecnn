import torch

from image_classifier import *
import random


# ImageClassifer defines the structure of the CNN; Trainer is used to actually train it
class Trainer:
    def __init__(self, data_iter, grayscale, learning_rate=0.001, momentum=0.9): # these constants are explained in later comments
        self.data_iter = data_iter
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Define the CNN to be trained
        if grayscale:
            self.image_classifier = GrayscaleImageClassifier() # note that Trainer is not a subclass of ImageClassifier, but instances of this class 'own' ImageClassifiers, which they train
        else:
            self.image_classifier = RGBImageClassifier()

        # Define training variables
        self.loss_fn = torch.nn.CrossEntropyLoss() # this function computes the difference between the CNN's answer and the labelled answer


        self.optimiser = torch.optim.SGD(self.image_classifier.parameters(), lr=learning_rate, momentum=momentum)
        """
            Stochastic Gradient Descent (SGD) uses backpropagation to slowly descend down the graph of the loss function to find a local minimum (this is called 'learning')
            The 'lr' (learning rate) corresponds to the amount that each weight/bias/filter adjusts during backpropagation
            A learning rate of 0.001 appears to result in the fastest and most effective learning
            Momentum refers to the smoothing of noisy data, and a visualisation, as well as a reinforcement that 0.9 is the optimal value, can be found at:
            https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
        """

        
        self.reset_accuracy()
        self.true_total = 0 # 'true_total' never changes, whereas 'total' can be reset
        self.softmax = torch.nn.Softmax(dim=1) # this converts the final layer of neurons into probabilities 0 <= p <= 1 which sum to 1
        self.delay = 0

        self.correct_guess = False

    def training_step(self):
        # Get the next image and label
        self.images, self.labels = self.data_iter.next() # the variable names are pluralised because 'images', for instance, is a list with one 'image' in it

        self.optimiser.zero_grad() # these gradients are used to compute backpropagation, but must be reset to 0 each loop to avoid compounding effects

        # Run a 'forward pass' through the CNN
        self.outputs = self.image_classifier(self.images) # calling the object as a function is implemented for subclasses of 'torch.nn.Module' and ultimately runs 'ImageClassifier.forward'

        # Compute loss between guessed outputs and true labels while storing data about the weights/biases/filters for backpropagation later
        self.loss = self.loss_fn(self.outputs, self.labels)

        # Compute backpropagation in the following two lines, due to the simplicity of PyTorch
        self.loss.backward() # generate gradient values based on the results of the forward pass through the network
        self.optimiser.step() # update parameters in the network based on these gradients

        # Generate probabilities for each class (e.g. 97.23% ship, 0.51% airplane, etc.)
        self.probabilities = self.softmax(self.outputs)

        # Update display variables (accuracy, total, etc.)
        self.best_guess = torch.max(self.outputs, 1)[1] # the max of the outputs (value, tensor) is the best guess, then [1] gets the tensor
        self.correct_guess = bool(self.labels == self.best_guess) # True or False depending on whether the network guessed correctly

        # Update counters
        if self.correct_guess:
            self.correct += 1
        self.total += 1
        self.true_total += 1

    def guess_images(self, images, delay=0):
        if self.delay <= 0:
            # run the outputs through the network, but don't bother computing gradients or optimising anything, simplifying the process
            self.outputs = self.image_classifier(images)
            self.loss = -1 # tell Pygame not to render loss by setting to something it could never naturally equal
            self.probabilities = self.softmax(self.outputs)
            self.best_guess = torch.max(self.outputs, 1)[1]
            self.correct_guess = False
            self.delay = delay # after guessing the image, refuse to do anything for the next 'delay' function calls
        self.delay -= 1
        if self.delay < 0:
            self.delay = 0

    def reset_accuracy(self):
        """
            This function is called when the user believes the network has improved, but its past results are dragging down its accuracy.
            After resetting the accuracy, it will keep the 'true_total' but restart its ongoing calculation of the accuracy percentage 
        """
        self.correct = 0
        self.total = 0

    def get_accuracy(self):
        """
            Return the accuracy by dividing correct by total.
            max(1, x) is used as self.total can reset to 0, so the accuracy will be equal to 0/1 = 0 in that case.
        """
        return 100 * self.correct / max(1, self.total)
