import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from models.reco.TCN import TemporalConvNet
from models.reco.SelfAttention import Decoder
import math


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=(2, 2)):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=2, stride=stride, bias=False))
        # self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=stride))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)

        return torch.cat(features, 1)


class AttentionLayer(nn.Module):
    def __init__(self, out_channels, ker_size=7):
        super(AttentionLayer, self).__init__()

        # spatial attention
        ker_size = ker_size - 1 if not ker_size % 2 else ker_size
        pad = (ker_size - 1) // 2
        self.conv_sa = nn.Conv2d(1, 1, ker_size, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # N, C, H, W
        # spatial attention
        se = x.mean(1, keepdim=True)  # N 1 H W
        se1 = self.sigmoid(self.conv_sa(se))
        y = x * se1 + x

        # channel attention
        se = y.mean(-1).mean(-1)  # N C
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=1024, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(4, 8, 8), compression=0.7,
                 num_init_features=512, bn_size=4, drop_rate=0,
                 num_classes=2704, efficient=False, is_english=False, is_TCN=True, is_transformer=True):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        self.features = nn.Sequential()
        self.is_TCN = is_TCN
        self.is_transformer = is_transformer
        if is_english:
            english_flag = 1
        else:
            english_flag = 0

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
        ]))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            self.features.add_module('att%d' % (i + 1), AttentionLayer(num_features))
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i < len(block_config) - 1 - english_flag:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            else:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression), stride=(2, 1))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.features.add_module('norm_final', nn.BatchNorm2d(int(num_features)))

        self.conv1d0 = nn.Sequential(nn.Conv1d(1046, 1024, 1, 1),
                                     nn.BatchNorm1d(1024),
                                     nn.ReLU(True))

        if self.is_TCN:
            self.TCN = TemporalConvNet(1024, [1024, 1024, 1024, 1024], 2, dropout=0)
        if self.is_transformer:
            self.position_encoding = PositionalEncoding()
            self.multihead1 = Decoder(1024, 1024, 16, 4, 0)

        self.classifier = nn.Linear(1024, num_classes)


    def forward(self, text_kernel_features):
        features = self.features(text_kernel_features)

        out = F.relu(features)
        out = out.permute(0, 3, 1, 2)
        out = torch.flatten(out, 2)
        out = out.permute(0, 2, 1)
        out = self.conv1d0(out)

        if self.is_TCN:
            out = self.TCN(out)
        out = out.permute(2, 0, 1)

        if self.is_transformer:
            out = self.position_encoding(out)
            out = self.multihead1(out)

        out = out.permute(1, 0, 2)
        out_char = self.classifier(out)
        return out_char


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    model = DenseNet(num_classes=2704).to(device)

    print("modules have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    from torchsummary import summary

    print(summary(model, input_size=(512, 32, 240), batch_size=-1))
