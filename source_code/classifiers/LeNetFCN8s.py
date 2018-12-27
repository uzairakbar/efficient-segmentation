import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.inception as i3
from .fcn32s import get_upsampling_weight

class LeNetFCN8s(nn.Module):

    def __init__(self, num_classes=1000, transform_input=False):
        super(LeNetFCN8s, self).__init__()
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
        self.Mixed_7a = i3.InceptionD(768)
        self.Mixed_7b = i3.InceptionE(1280)
        self.Mixed_7c = i3.InceptionE(2048)
        
        # OTHER GUYS
        self.score_fr8 = nn.Conv2d(288, num_classes, 1)
        self.score_fr16 = nn.Conv2d(768, num_classes, 1)
        self.score_fr32 = nn.Conv2d(2048, num_classes, 1)
        self.upscore32 = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, 14, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=9, bias=False)
        
        # some drops here
        self.dropA = nn.Dropout2d(p=0.2)
        self.dropB = nn.Dropout2d(p=0.2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):# or 
                torch.nn.init.xavier_normal_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                print("ABORT! ABORT! NOT FULLY CONVOLUTIONAL!!!")
                quit()
            elif isinstance(m, nn.BatchNorm2d):
                #print("Weird. I have got a batchnorm in here")
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

        # MY VALUES
        
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.3769369 / 0.5) + (0.6353146 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.36186826 / 0.5) + (0.6300146 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.36188436 / 0.5) + (0.52398586 - 0.5) / 0.5
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
        # 17 x 17 x 768
        x = self.Mixed_7a(x)                            # 5.65625 x 5.65625 x 1280
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)                            # 5.65625 x 5.65625 x 2048
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)                            # 5.65625 x 5.65625 x 2048
        
        x = self.dropB(x)               # DROPOUT!
        x32 = x
        # 8 x 8 x 2048
#         print("32s", x.shape)
        ### MY OWN SHIT
        
        x8 = self.score_fr8(x8)
        x16 = self.score_fr16(x16)
        x32 = self.score_fr32(x32)
        # NOT MINE NOT MINE
        x = x32
        x = self.upscore32(x)                         # 8.65625 x 8.65625 x 768         ## 14 x 14 x numC
#         print("deco1", x.shape)
        pad2_16 = int((x.size()[2] - x16.size()[2])/2)
        pad3_16 = int((x.size()[3] - x16.size()[3])/2)
        x = x[:, :, pad2_16:pad2_16+x16.size()[2], pad3_16:pad3_16+x16.size()[3]]
        x = x + x16
        x = self.upscore16(x)                        # 10.65625 x 10.65625 x 288        ## 28 x 28 x numC
#         print("deco2", x.shape)
        pad2_8 = int((x.size()[2] - x8.size()[2])/2)
        pad3_8 = int((x.size()[3] - x8.size()[3])/2)
        x = x[:, :, pad2_8:pad2_8+x8.size()[2], pad3_8:pad3_8+x8.size()[3]]
        x = x + x8
        x = self.upscore8(x)                         # 109.25 x 109.25 x num_classes    ## 224 x 224 x numC
#         print("deco3", x.shape)
        pad2_out = int((x.size()[2] - input.size()[2])/2)
        pad3_out = int((x.size()[3] - input.size()[3])/2)
        x = x[:, :, pad2_out:pad2_out+input.size()[2], pad3_out:pad3_out+input.size()[3]].contiguous()
#         print("output", x.shape)
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
                l2.load_state_dict(l1.state_dict())           

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
