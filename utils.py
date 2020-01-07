"""
This python file is used to store some functions and classes that will be used in the construction of the network.
"""
from torch import nn


def norm_conv(input_channel, output_channel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
        nn.PReLU(),
        nn.BatchNorm2d(output_channel))


class BottleNeck(nn.Module):
    """
    Basic BottleNeck for MobileFaceNet
    """

    def __init__(self, repeat, expension_factor, in_ch, out_ch, stride):
        super(BottleNeck, self).__init__()
        self.repeat = repeat
        self.expansion_factor = expension_factor
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.middle_out = 0

        self.modulist = self._net()

    def forward(self, x):
        output_list = []

        for i in range(self.repeat):
            if i == 0:
                out = self.modulist[i](x)
                output_list.append(out)
            else:
                out = self.modulist[i](output_list[i - 1])
                output_list.append(out)
                out = out + output_list[i - 1]
        # print(out.size())
        return out

    def _net(self):
        modulist = nn.ModuleList()
        model_all = nn.Sequential()
        kernel_size = [1, 3, 1]
        stride_size = [1, 2, 1]
        padding = [0, 1, 0]

        for j in range(self.repeat):
            model = nn.Sequential()

            # only the first layer need a stride of 2
            if j == 0:
                self.middle_out = self.out_ch

            if self.stride == 1:
                stride_size[1] = 1

            middle_ch = self.expansion_factor * self.in_ch
            input_ch = [self.in_ch, middle_ch, middle_ch]
            output_ch = [middle_ch, middle_ch, self.out_ch]

            group = [1, middle_ch, 1]

            for i in range(3):
                index = 3 * j + i + 1
                conv_2d = nn.Conv2d(input_ch[i], output_ch[i], kernel_size[i],
                                    stride_size[i], padding[i], groups=group[i])
                model.add_module(f'conv_{index}', conv_2d)
                model.add_module(f'PRelu{index}', nn.PReLU())
                if i == 2:
                    pass
                else:
                    model.add_module(f'BN{index}', nn.BatchNorm2d(output_ch[i]))
                model_all.add_module(f'model_{index}', model)

            modulist.append(model)

            kernel_size = [1, 1, 1]
            padding = [0, 0, 0]
            stride_size = [1, 1, 1]
            self.in_ch = self.out_ch
        return modulist
