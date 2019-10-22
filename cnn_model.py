import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
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
    """
    def __init__(self, n_a, width):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, width+4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(width+4, 2*(width+4), kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear((width//2)//2 *(width//2)//2 * 2*(width+4), 1000)
        self.fc2 = nn.Linear(1000, n_a)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out