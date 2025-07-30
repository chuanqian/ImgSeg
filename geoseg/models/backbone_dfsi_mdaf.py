import torch
from torch import nn
import timm
from torch.nn import functional as F
import numbers
from einops import rearrange


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp):
        x0 = self.conv(inp)
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)
        return out

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([conv_weight, conv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

        bias = torch.cat([conv_bias, conv_bn_bias], 0)
        bias = torch.matmul(weight_compress, bias)

        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias


class HierarchicalDualPathAttention(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.att1 = nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.att(x)
        max_out, _ = torch.max(x1 * x, dim=1, keepdim=True)
        x2 = self.att1(max_out)
        x3 = torch.mul(x2, x1) * x
        return x3


##动态滤波器的功能  对高 低频 进行权重分配和融合    在这里进行分组g的测试 可以。  2 4 8 16 32
class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=32):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  # 两个可学习的参数 分别对应高频和低频 用于动态调整滤波
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)  # 1*1卷积 用于生成 低频滤波器   g*k*k大小 输出
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)  # BN稳定训练
        self.act = nn.Softmax(dim=-2)  # 归一化 低频滤波器的权重
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size // 2)  # 对输入进行边界填充，保持输入张量的空间大小不变

        self.ap = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化  压缩空间维度 和 提取全局信息
        self.modulate = SFconv(inchannels)  ###进行调制

    def forward(self, x):
        identity_input = x  # 方便进行残差 链接
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)  # 全局平均池化 生成空间紧凑的低频特征 通过1*1卷积得到 动态滤波器 并归一化

        n, c, h, w = x.shape  # 将特征 转换成 滑动窗口模式 unfold函数  提取k*k的局部感受野 特征   n  g  c/g  K*k  hw
        # 每一组特征（self.group）分离出 c // self.group 通道，并展开为 kernel_size ** 2 滑动窗口特征，覆盖整个特征图的空间位置（h * w）。
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)  # 转化后的特征 表示每个组的 局部特征。
        # c1: 滤波器通道数。p, q: 滤波器的空间大小（通常是池化后的特征尺寸）。
        n, c1, p, q = low_filter.shape  # 调整 低通滤波器的形状  与展开后的 输入特征（滑动窗口的特征） 相匹配  并通过softmax归一化  确保滤波权重的 有效性。  n c/k*k k*k p*q
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(
            2)  # 在通道维度拆分为每个窗口特征的通道数 在第 2 维增加一个维度，用于匹配滑动窗口特征 x
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)  # 展开后的特征x  与低频滤波器 按照通道维度进行加权求和   在reshape回 nchw形状。

        out_high = identity_input - low_part  # 高频特征 通过 残差计算得到  原始特征减去低频特征
        out = self.modulate(low_part, out_high)  # 调用 SFconv 调制低频和高频的权重 并融合生成最后的输出特征。
        # print("low_part: ", low_part.shape)
        # print("out_high: ", out_high.shape)
        return out


##选择调制模块 Select Frequency Conv
class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)  # 这个FC操作是Conv 这是一个 1x1 卷积操作（虽然被称为 “FC”），它的作用类似于全连接层，用于减少通道数，从 features 压缩到 d。
        # 这里使用了M个1*1卷积层，每个卷积层对应一个频率组（如高频和低频）作用是从压缩后的特征维度d恢复到原始维度features。
        self.fcs = nn.ModuleList([])  # 也是Conv
        for i in range(M):  # M：通道划分的子组数（默认 2，表示两个子组：高频和低频）。
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)  # 用于归一化通道权重
        self.out = nn.Conv2d(features, features, 1, 1, 0)  # 对融合后的高频和低频特征进行调整，生成最终的输出
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, low, high):
        emerge = low + high  # sum在一起
        emerge = self.gap(emerge)  # GAP操作

        fea_z = self.fc(emerge)  # 分成两部分了  1*1 卷积 压缩通道 通过 self.fc 将特征通道数压缩到 d

        high_att = self.fcs[0](fea_z)  # 用 self.fcs 中的卷积层分别为高频和低频生成通道权重
        low_att = self.fcs[1](fea_z)

        attention_vectors = torch.cat([high_att, low_att], dim=1)  # 拼接 将高频和低频的权重拼接起来

        attention_vectors = self.softmax(attention_vectors)  # softmax 通过 Softmax 操作进行归一化，确保高频和低频的权重总和为 1
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)  # 分成两部分 将归一化后的权重拆分回高频权重和低频权重

        fea_high = high * high_att  # 高 低频 分别和权重注意力  高频特征乘以其对应的权重
        fea_low = low * low_att

        out = self.out(fea_high + fea_low)  # 融合 将加权后的高频和低频特征相加，形成融合特征   self.out 进一步调整输出特征
        return out


# --------------------------------------------------------------------------------

##基础卷积 conv+bn+GELU
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class MDAF(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type, ):
        super().__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_1_3 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1_2_3 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_1_3 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv2_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_2_3 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        attn_221 = self.conv2_2_1(x2)
        attn_222 = self.conv2_2_2(x2)
        attn_223 = self.conv2_2_3(x2)

        out1 = attn_111 + attn_112 + attn_113 + attn_121 + attn_122 + attn_123
        out2 = attn_211 + attn_212 + attn_213 + attn_221 + attn_222 + attn_223
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1 + x2
        return out


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class backbone_dfsi_mdaf(nn.Module):
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
        # self.hdpa = HierarchicalDualPathAttention(decode_channels, rep_scale=rep_scale)
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
        # channels_feat = self.hdpa(middle_feat)
        # 用空间小波变换处理空间
        spatial_feat = self.dfsi(middle_feat)
        feat = self.mdaf(middle_feat, spatial_feat)
        feat = self.wf(feat, res1)
        # feat = self.segmentation_head(feat)
        feat = self.segmentation_head(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


if __name__ == '__main__':
    model = backbone_dfsi_mdaf(num_classes=6, decode_channels=96)
    x = torch.randn(4, 3, 1024, 1024)
    out = model(x)
    print(out.shape)
