"""
This python file is used to build a mobilenet v2 to proceed images with an input of 112x112x1.
"""
import os
import torch
import time
import copy
from torch import nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils import BottleNeck, norm_conv


class MobileNet112(nn.Module):
    """
    input size: 1, 112 , 112
    The stride of First Convolutional Layer is 2.
    Network Levels: 7 Bottlenecks.
    """

    def __init__(self, category):
        """
        :param category:
        """
        super(MobileNet112, self).__init__()

        self.bn_x = nn.BatchNorm2d(1)

        self.norm_conv1 = norm_conv(1, 32, 3, 2, 1)
        self.bottleneck1 = BottleNeck(1, 1, 32, 16, 1)
        self.bottleneck2 = BottleNeck(2, 6, 16, 24, 1)
        self.bottleneck3 = BottleNeck(3, 6, 24, 32, 2)
        self.bottleneck4 = BottleNeck(4, 6, 32, 64, 2)
        self.bottleneck5 = BottleNeck(3, 6, 64, 96, 1)
        self.bottleneck6 = BottleNeck(3, 6, 96, 160, 2)
        self.bottleneck7 = BottleNeck(1, 6, 160, 320, 1)
        self.norm_conv2 = norm_conv(320, 960, 1, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.norm_conv3 = norm_conv(960, category, 1, 1)

    def forward(self, x):
        x = self.bn_x(x)

        x = self.norm_conv1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.norm_conv2(x)
        x = self.avgpool(x)
        x = self.norm_conv3(x)

        x = x.view(x.size(0), -1)
        return x


class MobileNet112V1(nn.Module):
    """
    Network Levels: 6 Bottlenecks(Removed the last bottleneck layer,
    and change the output of 6th bottleneck layer to 7*7*320
    input size: 1, 112 , 112
    The stride of First Convolutional Layer is 2.
    """

    def __init__(self, category):
        """
        :param category:
        """
        super(MobileNet112V1, self).__init__()

        self.bn_x = nn.BatchNorm2d(1)

        self.norm_conv1 = norm_conv(1, 32, 3, 2, 1)
        self.bottleneck1 = BottleNeck(1, 1, 32, 16, 1)
        self.bottleneck2 = BottleNeck(2, 6, 16, 24, 1)
        self.bottleneck3 = BottleNeck(3, 6, 24, 32, 2)
        self.bottleneck4 = BottleNeck(4, 6, 32, 64, 2)
        self.bottleneck5 = BottleNeck(3, 6, 64, 96, 1)
        self.bottleneck6 = BottleNeck(3, 6, 96, 320, 2)
        self.norm_conv2 = norm_conv(320, 960, 1, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.norm_conv3 = norm_conv(960, category, 1, 1)

    def forward(self, x):
        x = self.bn_x(x)

        x = self.norm_conv1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.norm_conv2(x)
        x = self.avgpool(x)
        x = self.norm_conv3(x)

        x = x.view(x.size(0), -1)
        return x


def data_transforms(data_dir='/home/charles/Documents/Dataset/FER2013/imgs', batch_size=128):
    data_transformers = {'train': transforms.Compose([transforms.RandomHorizontalFlip(),
                                                      transforms.Grayscale(),
                                                      transforms.ToTensor()]),
                         'val': transforms.Compose([transforms.Grayscale(),
                                                    transforms.ToTensor()])}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transformers[x])
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
    desired_best_acc = model_acc
    best_training_acc = 0.0

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
                # print(inputs)
                # print(inputs.size())
                # print(labels.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(labels)
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

            # get a new high accuracy
            if phase == 'val' and epoch_acc > best_training_acc:
                best_training_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # if training accuracy is better than best model, save the model.
            if desired_best_acc < best_training_acc:
                desired_best_acc = best_training_acc

                model.load_state_dict(best_model_wts)
                localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                new_model_name = model_name.split('.')[0][:28] + '_' + str(
                    100 * round(desired_best_acc.item(), 4)) + '%_' + localtime
                torch.save(model.state_dict(), new_model_name)
            # Update training or validation information.
            if phase == 'train':
                writer.add_scalar('training loss', epoch_loss, epoch + 1)
                writer.add_scalar('training accuracy', epoch_acc, epoch + 1)

            elif phase == 'val':
                writer.add_scalar('val loss', epoch_loss, epoch + 1)
                writer.add_scalar('val accuracy', epoch_acc, epoch + 1)

        time_elapse = time.time() - start
        print('This Epoch costs {:.0f}m {:.0f}s'.format(
            time_elapse // 60, time_elapse % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_training_acc))


if __name__ == '__main__':
    model_ft = MobileNet112(7).cuda()
    print(model_ft)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.1)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    # dataloaders, dataset_sizes = data_transforms(data_dir='/home/charles/DL/Badanamu/Facial_Expression/FER13/dataset',
    #                                              batch_size=64)
    dataloaders, dataset_sizes = data_transforms(data_dir='/home/charles/Documents/Dataset/FER2013/imgs', batch_size=64)

    train_model(model_ft, 0.50, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes,
                model_name="mobilenet_v2_2C_7B_F960_adam_89%_2019_09_26_12_0etet9_41", num_epochs=80)
