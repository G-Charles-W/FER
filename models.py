import torch
import torch.nn as nn
import numpy as np
# import torchsnooper
import os
from torchvision import datasets, models, transforms
import time
import copy
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


class MobileNet48(nn.Module):
    def __init__(self, category):
        """
        input size: 1, 48, 48
        :param category:
        """
        super(MobileNet48, self).__init__()
        # self.bn_x = nn.BatchNorm2d(1)
        # self.norm_conv1 = nn.Conv2d(3, 32, 3, 2, 1)

        self.norm_conv1 = self.norm_conv(1, 32, 3, 1, 1)

        self.bottleneck1 = BottleNeck(1, 1, 32, 16, 1)
        self.bottleneck2 = BottleNeck(2, 6, 16, 24, 1)
        self.bottleneck3 = BottleNeck(3, 6, 24, 32, 2)
        self.bottleneck4 = BottleNeck(4, 6, 32, 64, 2)
        self.bottleneck5 = BottleNeck(3, 6, 32, 96, 1)
        self.bottleneck6 = BottleNeck(3, 6, 96, 160, 2)
        self.bottleneck7 = BottleNeck(3, 6, 160, 320, 1)

        self.norm_conv2 = self.norm_conv(320, 960, 1, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.norm_conv3 = self.norm_conv(960, category, 1, 1)
        # self.norm_conv2 = nn.Conv2d(320, 1280, 1, 1)
        # self.avgpool = nn.AvgPool2d(7, 7)
        # self.norm_conv3 = nn.Conv2d(1280, category, 1, 1)

    def forward(self, x):
        # x = self.bn_x(x)

        x = self.norm_conv1(x)
        # print('norm_conv1', x.size())
        # x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        # x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)

        x = self.norm_conv2(x)
        # print('norm_conv2', x.size())
        x = self.avgpool(x)
        # print('avgpool', x.size())
        x = self.norm_conv3(x)
        # print('norm_conv3', x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        #
        # print(x)
        return x

    @staticmethod
    def depth_wise_conv2d(input_channel, output_channel, kernel_size, stride, padding):
        conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          groups=input_channel)
        conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0)
        model = nn.Sequential(conv1, conv2)
        return model

    @staticmethod
    def norm_conv(input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.PReLU(),
            nn.BatchNorm2d(output_channel))


class MobileNet2_7B48(nn.Module):
    def __init__(self, category):
        """
        input size: 1, 48, 48
        :param category:
        """
        super(MobileNet2_7B48, self).__init__()

        self.norm_conv1 = self.norm_conv(1, 32, 3, 1, 1)

        self.bottleneck1 = BottleNeck(1, 1, 32, 16, 1)
        self.bottleneck2 = BottleNeck(2, 6, 16, 24, 2)
        self.bottleneck3 = BottleNeck(3, 6, 24, 32, 1)
        self.bottleneck4 = BottleNeck(4, 6, 32, 64, 2)
        self.bottleneck5 = BottleNeck(3, 6, 64, 96, 1)
        self.bottleneck6 = BottleNeck(3, 6, 96, 160, 2)
        self.bottleneck7 = BottleNeck(3, 6, 160, 320, 1)

        self.norm_conv2 = self.norm_conv(320, 960, 1, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.norm_conv3 = self.norm_conv(960, category, 1, 1)
        # self.norm_conv2 = nn.Conv2d(320, 1280, 1, 1)
        # self.avgpool = nn.AvgPool2d(7, 7)
        # self.norm_conv3 = nn.Conv2d(1280, category, 1, 1)

    def forward(self, x):
        # x = self.bn_x(x)

        x = self.norm_conv1(x)
        # print(x.size())
        # print('norm_conv1', x.size())
        x = self.bottleneck1(x)
        # print(x.size())
        x = self.bottleneck2(x)
        # print(x.size())
        x = self.bottleneck3(x)
        # print(x.size())
        x = self.bottleneck4(x)
        # print(x.size())
        x = self.bottleneck5(x)
        # print(x.size())
        x = self.bottleneck6(x)
        # print(x.size())
        x = self.bottleneck7(x)
        # print(x.size())
        x = self.norm_conv2(x)
        # print(x.size())
        # print('norm_conv2', x.size())
        x = self.avgpool(x)
        # print(x.size())
        # print('avgpool', x.size())
        x = self.norm_conv3(x)
        # print(x.size())
        # print('norm_conv3', x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        #
        # print(x)
        return x

    @staticmethod
    def depth_wise_conv2d(input_channel, output_channel, kernel_size, stride, padding):
        conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          groups=input_channel)
        conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0)
        model = nn.Sequential(conv1, conv2)
        return model

    @staticmethod
    def norm_conv(input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.PReLU(),
            nn.BatchNorm2d(output_channel))


class MobileNet6B48(nn.Module):
    def __init__(self, category):
        """
        input size: 1, 48, 48
        :param category:
        """
        super(MobileNet6B48, self).__init__()
        self.bn_x = nn.BatchNorm2d(1)
        # self.norm_conv1 = nn.Conv2d(3, 32, 3, 2, 1)

        self.norm_conv1 = self.norm_conv(1, 32, 3, 1, 1)

        self.bottleneck1 = BottleNeck(1, 1, 32, 16, 1)
        self.bottleneck2 = BottleNeck(2, 6, 16, 24, 1)
        self.bottleneck3 = BottleNeck(3, 6, 24, 32, 2)
        self.bottleneck4 = BottleNeck(4, 6, 32, 64, 2)
        # self.bottleneck5 = BottleNeck(3, 6, 64, 96, 1)
        self.bottleneck6 = BottleNeck(3, 6, 64, 160, 2)
        self.bottleneck7 = BottleNeck(3, 6, 160, 320, 1)

        self.norm_conv2 = self.norm_conv(320, 1280, 1, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.norm_conv3 = self.norm_conv(1280, category, 1, 1)
        # self.norm_conv2 = nn.Conv2d(320, 1280, 1, 1)
        # self.avgpool = nn.AvgPool2d(7, 7)
        # self.norm_conv3 = nn.Conv2d(1280, category, 1, 1)

    def forward(self, x):
        x = self.bn_x(x)

        x = self.norm_conv1(x)
        # print('norm_conv1', x.size())
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        # x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)

        x = self.norm_conv2(x)
        # print('norm_conv2', x.size())
        x = self.avgpool(x)
        # print('avgpool', x.size())
        x = self.norm_conv3(x)
        # print('norm_conv3', x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        #
        # print(x)
        return x

    @staticmethod
    def depth_wise_conv2d(input_channel, output_channel, kernel_size, stride, padding):
        conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          groups=input_channel)
        conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0)
        model = nn.Sequential(conv1, conv2)
        return model

    @staticmethod
    def norm_conv(input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.PReLU(),
            nn.BatchNorm2d(output_channel))


class BottleNeck(nn.Module):
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


def data_transforms(data_dir='/home/charles/DL/Badanamu/Facial_Expression/FER13/dataset', batch_size=128):
    data_transforms = {'train': transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.Grayscale(),
                                                    transforms.ToTensor()]),
                       'val': transforms.Compose([transforms.Grayscale(),
                                                  transforms.ToTensor()])}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloaders, dataset_sizes


def train_model(model, model_acc, optimizer, scheduler, dataloaders, dataset_sizes, model_name, num_epochs=25):
    writer = SummaryWriter('runs/mobilenet_v2_2C_7B_F640_adam')
    since = time.time()
    # if os.path.exists("mobilenet_v2_best_model_adam.pth"):
    #     print(" the best model has been loaded")
    #     model.load_state_dict(torch.load("mobilenet_v2_best_model_adam.pth"))
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(model_name):
        print(f"The best model {model_name} has been loaded")
        model.load_state_dict(torch.load(model_name))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_model = model_acc
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #
                # print(f'inputs: {inputs}')
                # print(f'inputs size:{inputs.size()}')
                # print(f'labels:{labels}')
                # print(f'labels.size():{labels.size()}')

                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                print(inputs)
                print(inputs.size())
                print(labels.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    print(labels)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # load best model weights
            if best_acc_model < best_acc:
                best_acc_model = best_acc

                model.load_state_dict(best_model_wts)
                localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                new_model_name = model_name.split('.')[0][:28] + '_' + str(
                    100 * round(best_acc_model.item(), 4)) + '%_' + localtime
                torch.save(model.state_dict(), new_model_name)

            if phase == 'train':
                writer.add_scalar('training loss', epoch_loss, epoch + 1)
                writer.add_scalar('training accuracy', epoch_acc, epoch + 1)

            if phase == 'val':
                writer.add_scalar('val loss', epoch_loss, epoch + 1)
                writer.add_scalar('val accuracy', epoch_acc, epoch + 1)
        time_elapse = time.time() - start
        print('This Epoch costs {:.0f}m {:.0f}s'.format(
            time_elapse // 60, time_elapse % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


if __name__ == "__main__":
    # 2 class
    model_ft = MobileNet2_7B48(2).cuda()
    print(model_ft)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    # dataloaders, dataset_sizes = data_transforms(data_dir='/home/charles/DL/Badanamu/Facial_Expression/FER13/dataset',
    #                                              batch_size=64)
    dataloaders, dataset_sizes = data_transforms(
        data_dir='/home/charles/Badanamu/DL/Badanamu/Facial_Expression/2class/dataset',
        batch_size=64)

    train_model(model_ft, 0.89, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes,
                model_name="mobilenet_v2_2C_7B_F960_adam_89%_2019_09_26_12_09_41", num_epochs=80)

    # model_ft = MobileNet48(6).cuda()
    # print(model_ft)
    #
    # # Observe that all parameters are being optimized
    # # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.1)
    #
    # # Decay LR by a factor of 0.1 every 40 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    #
    # # dataloaders, dataset_sizes = data_transforms(data_dir='/home/charles/DL/Badanamu/Facial_Expression/FER13/dataset',
    # #                                              batch_size=64)
    # dataloaders, dataset_sizes = data_transforms(
    #     data_dir='/home/gene/DeepLearning/Badanamu/DL/Badanamu/Facial_Expression/FER13/dataset',
    #     batch_size=32)
    #
    # train_model(model_ft, 0.5191, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes,
    #             model_name="mobilenet_v2_6C_5B_F640_adam_51.91%_2019_09_26_17_40_48", num_epochs=20)

'''
6class training
model_ft = MobileNet48(2).cuda()
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.1)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    print(model_ft)
    dataloaders, dataset_sizes = data_transforms(data_dir='/home/charles/DL/Badanamu/Facial_Expression/FER13/dataset')
    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=80)
'''

'''
model test:
    x = torch.randn(2, 1, 48, 48)
    mobile_net = MobileNet48(2)
    print(mobile_net)
    y = mobile_net(x)
    print(y.size())
'''