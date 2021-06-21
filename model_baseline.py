import torch
import torch.nn as nn
import torch.nn.functional as F
#https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, rate=1, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      stride=stride, dilation = rate, 
                      padding=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                      kernel_size=3, dilation = rate, 
                      padding=rate, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ATpooling(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        rates = [1,4,8,12]
        self.ATconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=1, stride=stride, 
                                 dilation = rates[0], padding=0, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace = True))
        self.ATconv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, stride=stride, 
                                 dilation = rates[1], padding=rates[1], bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace = True))
        self.ATconv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, stride=stride, 
                                 dilation = rates[2], padding=rates[2], bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace = True))
        self.ATconv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, stride=stride, 
                                 dilation = rates[3], padding=rates[3], bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace = True))
        self.ATconv5 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace = True))
        self.convf = nn.Conv2d(in_channels = out_channels * 5, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride=1,
                          padding = 0,
                          dilation=1,
                          bias=False)
    def forward(self, x):
        x1 = self.ATconv1(x)
        x2 = self.ATconv2(x)
        x3 = self.ATconv3(x)
        x4 = self.ATconv4(x)
        x5 = F.interpolate(self.ATconv5(x), size = tuple(x4.shape[-2:]), mode='bilinear')
        return torch.cat([x1, x2, x3, x4, x5], dim = 1)
                          

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, rate=1, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, dilation = rate, padding=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class MyrealDeepLabV3_4(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride = 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.Blocks_1 = nn.Sequential(BasicBlock(64, 64, rate = 1, stride = 1),
                                      BasicBlock(64, 64, rate = 1, stride = 1),
                                      BasicBlock(64, 128, rate = 1, stride = 2),
                                      BasicBlock(128, 128, rate = 1, stride = 1),
                                      BasicBlock(128, 128, rate = 1, stride = 1),
                                      BasicBlock(128, 128, rate = 1, stride = 1))
                                      
        self.Blocks_2 = nn.Sequential(BasicBlock(128, 256, rate = 1, stride = 2),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 2, stride = 1))
        self.ATpooling = ATpooling(256, 256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256*5, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.UPscale1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 2, stride=2),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.UPscale2 = nn.Sequential(nn.Conv2d(384, 384, 3, 1, 1),
                                      nn.BatchNorm2d(384),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(384, 256, 8, stride=8),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
                                      
        self.final = nn.Sequential(nn.Conv2d(256, 80, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(80),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(80, 10, kernel_size=1, bias=False))
        

    def forward(self, x):
        x = self.conv1(x)
        x_prime = self.Blocks_1(x)
        x = self.Blocks_2(x_prime)
        x = self.ATpooling(x)
        x = self.conv2(x)
        x = self.UPscale1(x)
        x = self.UPscale2(torch.cat([x, x_prime], 1))
        x = self.final(x)
        return x
class MyrealDeepLabV3_4_universal(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride = 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.Blocks_1 = nn.Sequential(BasicBlock(64, 64, rate = 1, stride = 1),
                                      BasicBlock(64, 64, rate = 1, stride = 1),
                                      BasicBlock(64, 128, rate = 1, stride = 2),
                                      BasicBlock(128, 128, rate = 1, stride = 1),
                                      BasicBlock(128, 128, rate = 1, stride = 1),
                                      BasicBlock(128, 128, rate = 1, stride = 1))
                                      
        self.Blocks_2 = nn.Sequential(BasicBlock(128, 256, rate = 1, stride = 2),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 1, stride = 1),
                                      BasicBlock(256, 256, rate = 2, stride = 1))
        self.ATpooling = ATpooling(256, 256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256*5, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.UPscale1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 2, stride=2),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.UPscale2 = nn.Sequential(nn.Conv2d(384, 384, 3, 1, 1),
                                      nn.BatchNorm2d(384),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(384, 256, 8, stride=8),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
                                      
        self.final = nn.Sequential(nn.Conv2d(256, 80, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(80),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(80, 10, kernel_size=1, bias=False))

    def forward(self, x):
        input_size = x.size()
        x = self.conv1(x)
        x_prime = self.Blocks_1(x)
        x = self.Blocks_2(x_prime)
        x = self.ATpooling(x)
        x = self.conv2(x)
        x = self.UPscale1(x)
        x = F.interpolate(x, size = tuple(x_prime.shape[-2:]), mode='bilinear')
        x = self.UPscale2(torch.cat([x, x_prime], 1))
        x = F.interpolate(x, size = tuple(input_size[-2:]), mode='bilinear')
        x = self.final(x)
        return x
class MyrealDeepLabV3_5(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride = 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.Blocks_1 = nn.Sequential(BottleNeck(64, 64, rate = 1, stride = 1),
                                      BottleNeck(256, 64, rate = 1, stride = 1),
                                      BottleNeck(256, 64, rate = 1, stride = 2),
                                      BottleNeck(256, 128, rate = 1, stride = 1),
                                      BottleNeck(512, 128, rate = 1, stride = 1),
                                      BottleNeck(512, 128, rate = 1, stride = 1))
                                      
        self.Blocks_2 = nn.Sequential(BottleNeck(512, 256, rate = 1, stride = 2),
                                      BottleNeck(1024, 256, rate = 1, stride = 1),
                                      BottleNeck(1024, 256, rate = 1, stride = 1),
                                      BottleNeck(1024, 256, rate = 1, stride = 1),
                                      BottleNeck(1024, 256, rate = 1, stride = 1),
                                      BottleNeck(1024, 256, rate = 2, stride = 1))
        self.ATpooling = ATpooling(1024, 1024)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024*5, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))
        self.UPscale1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 2, stride=2),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))
        self.UPscale2 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(512, 256, 8, stride=8),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
                                      
        self.final = nn.Sequential(nn.Conv2d(256, 80, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(80),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(80, 10, kernel_size=1, bias=False))
        

    def forward(self, x):
        x = self.conv1(x)
        x_prime = self.Blocks_1(x)
        x = self.Blocks_2(x_prime)
        x = self.ATpooling(x)
        x = self.conv2(x)
        x = self.UPscale1(x)
        x = self.UPscale2(torch.cat([x, x_prime], 1))
        x = self.final(x)
        return x
class segnet(nn.Module):
    def __init__(self, out_channel=10):
        super(segnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)



class UNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        center = self.center(dec3)
        enc3 = self.enc3(torch.cat([
            center, F.interpolate(dec3, center.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.interpolate(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.interpolate(dec1, enc2.size()[2:])], 1))

        return F.interpolate(self.final(enc1), x.size()[2:])


if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = segnet()
    output = model(batch)
    print(output.size())