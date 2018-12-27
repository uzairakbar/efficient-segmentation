import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.inception as i3
from .fcn32s import get_upsampling_weight


# __all__ = ['Inception3', 'inception_v3']


# model_urls = {
#     # Inception v3 ported from TensorFlow
#     'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
# }


# def inception_v3(pretrained=False, **kwargs):
#     r"""Inception v3 model architecture from
#     `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         if 'transform_input' not in kwargs:
#             kwargs['transform_input'] = True
#         model = Inception3(**kwargs)
#         model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
#         return model

#     return Inception3(**kwargs)


class LeNetFCN8s(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(LeNetFCN8s, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = i3.BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = i3.BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = i3.BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = i3.BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = i3.BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = i3.InceptionA(192, pool_features=32)
        self.Mixed_5c = i3.InceptionA(256, pool_features=64)
        self.Mixed_5d = i3.InceptionA(288, pool_features=64)
        self.Mixed_6a = i3.InceptionB(288)
        self.Mixed_6b = i3.InceptionC(768, channels_7x7=128)
        self.Mixed_6c = i3.InceptionC(768, channels_7x7=160)
        self.Mixed_6d = i3.InceptionC(768, channels_7x7=160)
        self.Mixed_6e = i3.InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = i3.InceptionAux(768, num_classes)
        self.Mixed_7a = i3.InceptionD(768)
        self.Mixed_7b = i3.InceptionE(1280)
        self.Mixed_7c = i3.InceptionE(2048)
        
        # OTHER GUYS
        self.score_fr8 = nn.Conv2d(288, num_classes, 4, padding=3)
        self.score_fr16 = nn.Conv2d(768, num_classes, 1, padding=1)
        self.score_fr32 = nn.Conv2d(2048, num_classes, 1, padding=1)
        self.upscore32 = nn.ConvTranspose2d(num_classes, num_classes, 8, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=2, padding=3, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4, bias=False)
        # self.fourthDeconv = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=8)
        
        # some drops here
        self.dropA = nn.Dropout2d(p=0.2)
        self.dropB = nn.Dropout2d(p=0.2)
        
        # MINE MINE MINE
#         self.score_fr = nn.Conv2d(2048, num_classes, 1)
#         self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
#                                           bias=False)
        
        
        # self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):# or 
                torch.nn.init.xavier_normal_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                print("ABORT! ABORT! NOT FULLY CONVOLUTIONAL!!!")
                quit()
            elif isinstance(m, nn.BatchNorm2d):
                print("Weird. I have got a batchnorm in here")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, input):
        x = input           # 224 x 224 x 3
#         print("input size", x.shape)
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)                       # 111.5 x 111.5 x 32
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)                       # 109.5 x 109.5 x 32
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)                       # 109.5 x 109.5 x 64
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)    # 54.25 x 54.25 x 64
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)                       # 54.25 x 54.25 x 80
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)                       # 52.25 x 52.25 x 192
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)    # 25.625 x 25.625 x 192
        # 35 x 35 x 192
        x = self.Mixed_5b(x)                            # 25.625 x 25.625 x 256
        # 35 x 35 x 256
        x = self.Mixed_5c(x)                            # 25.625 x 25.625 x 288
        x8 = x
        # 35 x 35 x 288
#         print("8s", x.shape)
        x = self.Mixed_5d(x)                            # 25.625 x 25.625 x 288
        # 35 x 35 x 288
        x = self.Mixed_6a(x)                            # 12.3125 x 12.3125 x 768
        # 17 x 17 x 768
        x = self.Mixed_6b(x)                            # 12.3125 x 12.3125 x 768
        # 17 x 17 x 768
        x = self.Mixed_6c(x)                            # 12.3125 x 12.3125 x 768
        # 17 x 17 x 768
        x = self.Mixed_6d(x)                            # 12.3125 x 12.3125 x 768
        # 17 x 17 x 768
        x = self.Mixed_6e(x)                            # 12.3125 x 12.3125 x 768
        x = self.dropA(x)               # DROPOUT!
        x16 = x
        # 17 x 17 x 768
#         print("16s", x.shape)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)                            # 5.65625 x 5.65625 x 1280
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)                            # 5.65625 x 5.65625 x 2048
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)                            # 5.65625 x 5.65625 x 2048
        
        x = self.dropB(x)               # DROPOUT!
        # 8 x 8 x 2048
#         print("32s", x.shape)
        ### x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        ### x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        ### x = x.view(x.size(0), -1)   SKIP THIS SHIT
        ## 2048                         SKIP THIS SHIT
        ### x = self.fc(x)
        ### MY OWN SHIT
        
        x8 = self.score_fr8(x8)
        x16 = self.score_fr16(x16)
        x32 = self.score_fr32(x)
        # NOT MINE NOT MINE
        x = x32
        x = self.upscore32(x)                         # 8.65625 x 8.65625 x 768         ## 14 x 14 x numC
#         print("deco1", x.shape)
        x = x + x16
        x = self.upscore16(x)                        # 10.65625 x 10.65625 x 288        ## 28 x 28 x numC
#         print("deco2", x.shape)
        x = x + x8
        x = self.upscore8(x)                         # 109.25 x 109.25 x num_classes    ## 224 x 224 x numC
#         print("deco3", x.shape)
        
        # MINE MINE
#         x = self.score_fr(x)
#         x = self.upscore(x)
        # x = x[:, :, 19:19 + input.size()[2], 19:19 + input.size()[3]].contiguous()
        
        # 1000 (num_classes)
        #if self.training and self.aux_logits:
        #    return x, aux
        return x
    
    def copy_params_from_leNet(self, leNet):
        selfFeatures = [
            self.Conv2d_1a_3x3,
            self.Conv2d_2a_3x3,
            self.Conv2d_2b_3x3,
            self.Conv2d_3b_1x1,
            self.Conv2d_4a_3x3,
            self.Mixed_5b,
            self.Mixed_5c,
            self.Mixed_5d,
            self.Mixed_6a,
            self.Mixed_6b,
            self.Mixed_6c,
            self.Mixed_6d,
            self.Mixed_6e,
#             self.AuxLogits,
            self.Mixed_7a,
            self.Mixed_7b,
            self.Mixed_7c
        ]
        
        leNetFeatures = [
            leNet.Conv2d_1a_3x3,
            leNet.Conv2d_2a_3x3,
            leNet.Conv2d_2b_3x3,
            leNet.Conv2d_3b_1x1,
            leNet.Conv2d_4a_3x3,
            leNet.Mixed_5b,
            leNet.Mixed_5c,
            leNet.Mixed_5d,
            leNet.Mixed_6a,
            leNet.Mixed_6b,
            leNet.Mixed_6c,
            leNet.Mixed_6d,
            leNet.Mixed_6e,
#             leNet.AuxLogits,
            leNet.Mixed_7a,
            leNet.Mixed_7b,
            leNet.Mixed_7c
        ]
        
        for l1, l2 in zip(leNetFeatures, selfFeatures):
            if isinstance(l1, type(l2)):
#                 assert l1.parameters().size() == l2.parameters().size()
                l2.load_state_dict(l1.state_dict())           
#                 l2.weight.data.copy_(l1.weight.data)
#                 l2.bias.data.copy_(l1.bias.data)
        for l1, l2 in zip([leNet.fc], [self.score_fr32]):
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


# class InceptionA(nn.Module):

#     def __init__(self, in_channels, pool_features):
#         super(InceptionA, self).__init__()
#         self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

#         self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
#         self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

#         self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
#         self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

#         self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch5x5 = self.branch5x5_1(x)
#         branch5x5 = self.branch5x5_2(branch5x5)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionB(nn.Module):

#     def __init__(self, in_channels):
#         super(InceptionB, self).__init__()
#         self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

#         self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
#         self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

#     def forward(self, x):
#         branch3x3 = self.branch3x3(x)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

#         branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

#         outputs = [branch3x3, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionC(nn.Module):

#     def __init__(self, in_channels, channels_7x7):
#         super(InceptionC, self).__init__()
#         self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

#         c7 = channels_7x7
#         self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
#         self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

#         self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
#         self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

#         self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch7x7 = self.branch7x7_1(x)
#         branch7x7 = self.branch7x7_2(branch7x7)
#         branch7x7 = self.branch7x7_3(branch7x7)

#         branch7x7dbl = self.branch7x7dbl_1(x)
#         branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
#         branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionD(nn.Module):

#     def __init__(self, in_channels):
#         super(InceptionD, self).__init__()
#         self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
#         self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

#         self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
#         self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
#         self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
#         self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

#     def forward(self, x):
#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = self.branch3x3_2(branch3x3)

#         branch7x7x3 = self.branch7x7x3_1(x)
#         branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
#         branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
#         branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

#         branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
#         outputs = [branch3x3, branch7x7x3, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionE(nn.Module):

#     def __init__(self, in_channels):
#         super(InceptionE, self).__init__()
#         self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

#         self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
#         self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
#         self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

#         self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
#         self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
#         self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

#         self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = [
#             self.branch3x3_2a(branch3x3),
#             self.branch3x3_2b(branch3x3),
#         ]
#         branch3x3 = torch.cat(branch3x3, 1)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = [
#             self.branch3x3dbl_3a(branch3x3dbl),
#             self.branch3x3dbl_3b(branch3x3dbl),
#         ]
#         branch3x3dbl = torch.cat(branch3x3dbl, 1)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)


# class InceptionAux(nn.Module):

#     def __init__(self, in_channels, num_classes):
#         super(InceptionAux, self).__init__()
#         self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
#         self.conv1 = BasicConv2d(128, 768, kernel_size=5)
#         self.conv1.stddev = 0.01
#         self.fc = nn.Linear(768, num_classes)
#         self.fc.stddev = 0.001

#     def forward(self, x):
#         # 17 x 17 x 768
#         x = F.avg_pool2d(x, kernel_size=5, stride=3)
#         # 5 x 5 x 768
#         x = self.conv0(x)
#         # 5 x 5 x 128
#         x = self.conv1(x)
#         # 1 x 1 x 768
#         x = x.view(x.size(0), -1)
#         # 768
#         x = self.fc(x)
#         # 1000
#         return x


# class BasicConv2d(nn.Module):

#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return F.relu(x, inplace=True)
