import torch
import torch.nn as nn
import torch.nn.functional as F


class GenreNet(nn.Module):

    def __init__(self, n_classes, input_shape=(1, 128, 1291)):
        super(GenreNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size=4)

        self.fc1 = nn.Linear(in_features=(512 * 3), out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=n_classes)

    @staticmethod
    def _relu_pool_drop(conv_layer, kernel_size):
        return F.dropout(F.max_pool1d(F.relu(conv_layer), kernel_size), p=0.25)

    def forward(self, inp):
        x = self._relu_pool_drop(self.conv1(inp), kernel_size=4)
        x = self._relu_pool_drop(self.conv2(x), kernel_size=2)
        x = self._relu_pool_drop(self.conv3(x), kernel_size=2)

        # Global temporal pooling
        operations = [
            F.avg_pool1d(x, kernel_size=x.size(2)),
            F.max_pool1d(x, kernel_size=x.size(2)),
            F.lp_pool2d(x, norm_type=2, kernel_size=(1, x.size(2)))
        ]
        x = torch.cat(operations, 1)

        x = x.view(1, 1, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x)
