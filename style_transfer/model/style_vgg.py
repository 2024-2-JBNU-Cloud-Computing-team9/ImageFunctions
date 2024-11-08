import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

## preprocess, postprocess
def preprocess(img_dir): # HxWxC or CxHxW -> BxCxHxW (B=1)
    img = Image.open(img_dir)
    transform = transforms.Compose( [transforms.ToTensor()] )
    img = transform(img).view((-1, 3, img.height, img.width))
    return img # tensor

def postprocess(tensor):  # BxCxHxW -> HxWxC
    img = tensor.clone()
    img = img.clamp(0,1)
    img = torch.squeeze(img, dim=0)
    img = torch.transpose(img, 0,1)
    img = torch.transpose(img, 1,2)
    return img


# 최신 권장 방식으로 VGG16 모델 불러오기
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)


## models
# vgg16 = models.vgg16(pretrained=True)

class VGG_FEATURE(nn.Module):
    def __init__(self):
        super(VGG_FEATURE,self).__init__()
        self.block1 = nn.Sequential(*list(vgg16.features.children())[0:3])
        self.block2 = nn.Sequential(*list(vgg16.features.children())[3:8])
        self.block3 = nn.Sequential(*list(vgg16.features.children())[8:15])
        self.block4 = nn.Sequential(*list(vgg16.features.children())[15:22])
        self.block5 = nn.Sequential(*list(vgg16.features.children())[22:29])

    def forward(self,x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        return out1, out2, out3, out4, out5
    
# 

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b,c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        return G
    
class GramMSELoss(nn.Module):
    def forward(self, input, target): # input: feature map, target: target gram matrix
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out
    
    
def make_net(device):
    vgg_feature = VGG_FEATURE()
    for param in vgg_feature.parameters():
        param.requires_grad = False
        
    return vgg_feature.to(device)