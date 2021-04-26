import torch
from torch.nn import ModuleList
import torch.nn as nn
import torchvision.models as models
from resnet import resnet34

class Resnet34_8s(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet34_8s, self).__init__()
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        # Randomly initialize the 1x1 Conv scoring laye
        self.resnet34_8s = nn.Sequential(*list(resnet34_8s.children())[:-2])
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)

        self.avg_pool_classifiers = ModuleList(2*[nn.AdaptiveAvgPool2d(output_size=(1,1))])
        self.linear_classifiers = ModuleList(2*[nn.Linear(in_features=512, out_features=1, bias=True)])
        self._normal_initialization(self.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.resnet34_8s(x)
        heatmap_pooled = self.avg_pool_1(x)
        heatmap = self.fc(heatmap_pooled)
        heatmap = nn.functional.upsample_bilinear(input=heatmap, size=input_spatial_dim)

        # z rotation
        cls_pooled_z = self.avg_pool_classifiers[0](x)
        cls_pooled_z = cls_pooled_z.view(cls_pooled_z.shape[0], cls_pooled_z.shape[1])
        cls_z = self.linear_classifiers[0](cls_pooled_z)

        # y rotation
        cls_pooled_y = self.avg_pool_classifiers[1](x)
        cls_pooled_y = cls_pooled_y.view(cls_pooled_y.shape[0], cls_pooled_y.shape[1])
        cls_y = self.linear_classifiers[1](cls_pooled_y)

        return heatmap, torch.squeeze(torch.stack((cls_z,), dim=1))
