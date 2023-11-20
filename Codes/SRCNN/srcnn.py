from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
