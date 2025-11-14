"""Improved model architectures with feature selection and attention mechanisms."""

import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation)."""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Average pool and max pool
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionResNet18(nn.Module):
    """ResNet18 with CBAM attention modules for better feature selection."""

    def __init__(self, num_classes=1, pretrained=False):
        super(AttentionResNet18, self).__init__()
        # Load base ResNet18
        self.model = models.resnet18(weights=None)

        # Adapt to grayscale input
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Add CBAM attention after each residual block
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # Replace the classification head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # Initial convolution
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Layer 1 + attention
        x = self.model.layer1(x)
        x = self.cbam1(x)

        # Layer 2 + attention
        x = self.model.layer2(x)
        x = self.cbam2(x)

        # Layer 3 + attention
        x = self.model.layer3(x)
        x = self.cbam3(x)

        # Layer 4 + attention
        x = self.model.layer4(x)
        x = self.cbam4(x)

        # Global pooling and classification
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 adapted for grayscale X-ray classification."""

    def __init__(self, num_classes=1, pretrained=False):
        super(EfficientNetB0, self).__init__()
        # Load EfficientNet-B0
        from torchvision.models import efficientnet_b0

        self.model = efficientnet_b0(weights=None)

        # Adapt first conv for grayscale
        self.model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class MultiScaleNet(nn.Module):
    """Multi-scale feature extraction network."""

    def __init__(self, num_classes=1):
        super(MultiScaleNet, self).__init__()
        self.model = models.resnet18(weights=None)

        # Adapt to grayscale
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Feature fusion
        self.fusion = nn.Conv2d(192, 64, kernel_size=1)

        # Classification head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # Initial features
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        # Multi-scale features
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)

        # Fuse multi-scale features
        x = torch.cat([s1, s2, s3], dim=1)
        x = self.fusion(x)

        # Continue with ResNet layers
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x


class DenseNet121(nn.Module):
    """DenseNet-121 for better feature reuse."""

    def __init__(self, num_classes=1):
        super(DenseNet121, self).__init__()
        from torchvision.models import densenet121

        self.model = densenet121(weights=None)

        # Adapt for grayscale
        self.model.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def get_model(model_name="attention_resnet18", **kwargs):
    """Factory function to get different model architectures.

    Args:
        model_name: One of ['attention_resnet18', 'efficientnet_b0',
                    'multiscale', 'densenet121', 'resnet18']

    Returns:
        Model instance
    """
    models_dict = {
        "attention_resnet18": AttentionResNet18,
        "efficientnet_b0": EfficientNetB0,
        "multiscale": MultiScaleNet,
        "densenet121": DenseNet121,
    }

    if model_name == "resnet18":
        # Return original ResNet18 for compatibility
        from berlin25_xray.task import Net

        return Net()

    model_class = models_dict.get(model_name)
    if model_class is None:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}"
        )

    return model_class(**kwargs)
