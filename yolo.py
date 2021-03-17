import torch
from torch import nn

class Yolo(nn.Module):
    def __init__(self, front, S = 7, B = 2, Class_num = 20):
        super(Yolo, self).__init__()
        self.S = S
        self.B = B
        self.Class_num = Class_num
        self.front = front
        self.net = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            #nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(1024, 1024, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            #nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            #nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            #nn.LeakyReLU(0.1, inplace = True)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.ReLU(inplace = True),
            #nn.LeakyReLU(0.1, inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, self.S * self.S * (5 * B + self.Class_num)),
            #nn.LeakyReLU(0.1, inplace = True)
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.front(x)
        x = self.net(x)
        x = self.fc(x)
        return x.view(x.shape[0], self.S, self.S, self.Class_num + 5 * self.B)
