import torch
import torch.nn as nn
import torch.nn.functional as F


class my_channel_attention(nn.Module):
    def __init__(self, channle):
        ratio = 16
        super(my_channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channle, channle//ratio),
            nn.ReLU(),
            nn.Linear(channle // ratio, channle)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = self.sigmoid(max_fc_out + avg_fc_out).view([b, c, 1, 1])
        return out*x


class my_spacial_atteition(nn.Module):
    def __init__(self, kernel_size=7):
        super(my_spacial_atteition, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out = torch.max(x, dim=1, keepdim=True).values
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        out = self.sigmoid(self.conv(pool_out))
        return out*x


class my_CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super(my_CBAM, self).__init__()
        self.channel_attention = my_channel_attention(c1)
        self.spacial_atteition = my_spacial_atteition(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_atteition(x)
        return x


# elif m in (CBAM, my_CBAM):
#     c1, c2 = ch[f], args[0]
#     if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
#         c2 = make_divisible(min(c2, max_channels) * width, 8)
#     args = [c1, *args[1:]]


        # elif m in (CBAM, my_CBAM):
        #     c1, c2 = ch[f], args[0]
        #     if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
        #         c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     args = [c1, *args[1:]]
