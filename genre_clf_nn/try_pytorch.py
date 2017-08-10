import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class GenreClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels = 1,
                               out_channels = 3,
                               kernel_size=4)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.drop1 = nn.Dropout(0.25)

        #x = torch.autograd.Variable(torch.FloatTensor([[[1, 2, 3, 4]]]))
        #c = nn.Conv1d(1, 1, 2, padding=1)
        #out = c(x)[:, :, :-1]

        self.conv2 = nn.Conv1d(in_channels = 1,
                               out_channels = 3,
                               kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels = 1,
                               out_channels = 3,
                               kernel_size=4)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop3 = nn.Dropout(0.25)
