import torch
from torch import nn
import timm
from torch.nn import functional as F
from geoseg.models.ConvBase import ConvBN, WF, ConvBNReLU, Conv
from geoseg.models.learnable_frequency_modeling import HierarchicalDualPathAttention, dynamic_filter, MDAF


class FrequencyModelNetwork(nn.Module):
    def __init__(self,
                 num_classes=6,
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=True,
                 decode_channels=96,
                 rep_scale=4,
                 dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained, output_stride=32,
                                          out_indices=(0, 1, 2, 3))
        encoder_channels = self.backbone.feature_info.channels()
        self.conv2 = ConvBN(encoder_channels[1], decode_channels, kernel_size=1)
        self.conv3 = ConvBN(encoder_channels[2], decode_channels, kernel_size=1)
        self.conv4 = ConvBN(encoder_channels[3], decode_channels, kernel_size=1)
        self.concat_conv1 = ConvBN(3 * decode_channels, decode_channels, kernel_size=1)
        self.hdpa = HierarchicalDualPathAttention(decode_channels, rep_scale=rep_scale)
        self.dfsi = dynamic_filter(decode_channels)
        self.mdaf = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.wf = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))

    def forward(self, x):
        b = x.size()[0]
        h, w = x.size()[-2:]

        res1, res2, res3, res4 = self.backbone(x)
        res1h, res1w = res1.size()[-2:]

        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bilinear', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bilinear', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bilinear', align_corners=False)
        middle_feat = self.concat_conv1(torch.cat([res2, res3, res4], dim=1))
        # 处理通道的
        channels_feat = self.hdpa(middle_feat)
        # 用空间小波变换处理空间
        spatial_feat = self.dfsi(middle_feat)
        feat = self.mdaf(channels_feat, spatial_feat)
        feat = self.wf(feat, res1)
        # feat = self.segmentation_head(feat)
        feat = self.segmentation_head(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


if __name__ == '__main__':
    model = FrequencyModelNetwork(num_classes=6, decode_channels=96)
    x = torch.randn(4, 3, 1024, 1024)
    out = model(x)
    print(out.shape)
