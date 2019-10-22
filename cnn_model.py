import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):

    def __init__(self, action_space_size, img_width, img_height):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)    #image trop petite
        #self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(img_width) #a modif selon le nbre de conv   conv2d_size_out(conv2d_size_out(img_width)) etc
        convh =conv2d_size_out(img_height)
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, action_space_size)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = F.relu(x)
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        #print(x.shape)
        #return self.head(x)
        return self.head(x.view(x.size(0), -1))