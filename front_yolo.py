import torch
from torch import nn

class FrontYolo(nn.Module):
    def __init__(self):
        super(FrontYolo, self).__init__()
        self.conv_net = self.make_conv_net()
        self.flatten = nn.Flatten()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(1024, 1000)

    def make_conv_net(self):
        ret = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),  #1
            nn.LeakyReLU(0.1, inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 1), #2
            nn.LeakyReLU(0.1, inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(192, 128, kernel_size = 1, stride = 1), #3
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), #4
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(256, 256, kernel_size = 1, stride = 1), #5
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1), #6
            nn.LeakyReLU(0.1, inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(512, 256, kernel_size = 1, stride = 1), #7
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1), #8
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1), #9
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1), #10
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1), #11
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1), #12
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1), #13
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1), #14
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(512, 512, kernel_size = 1, stride = 1), #15
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1), #16
            nn.LeakyReLU(0.1, inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1), # 17
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1), #18
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1), # 19
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1), #20
            nn.LeakyReLU(0.1, inplace = True)
        )
        return ret

    def forward(self, x):
        out = self.conv_net(x)
        out = self.avg(out)
        out = self.flatten(out)
        out = torch.squeeze(out)
        out = self.fc(out)
        return out
