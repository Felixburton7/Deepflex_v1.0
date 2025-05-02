"""
CNN models for Voxelflex.

This module contains PyTorch 3D CNN architectures for RMSF prediction.
"""

from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from voxelflex.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ResidualBlock3D(nn.Module):
    """3D Residual block with dilated convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        dropout_rate: float = 0.3
    ):
        """
        Initialize a 3D residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dilation: Dilation rate for convolution
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, 
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, 
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


class VoxelFlexCNN(nn.Module):
    """
    Basic 3D CNN architecture for RMSF prediction.
    
    This model uses a series of 3D convolutional layers followed by fully connected
    layers to predict RMSF values from voxelized protein data.
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        channel_growth_rate: float = 1.5,
        dropout_rate: float = 0.3
    ):
        """
        Initialize VoxelFlexCNN.
        
        Args:
            input_channels: Number of input channels (typically 4 or 5)
            base_filters: Number of filters in the first convolutional layer
            channel_growth_rate: Growth rate for channels in successive layers
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Calculate channel sizes for each layer
        c1 = base_filters
        c2 = int(c1 * channel_growth_rate)
        c3 = int(c2 * channel_growth_rate)
        c4 = int(c3 * channel_growth_rate)
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(c2, c3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(c3, c4, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm3d(c1)
        self.bn2 = nn.BatchNorm3d(c2)
        self.bn3 = nn.BatchNorm3d(c3)
        self.bn4 = nn.BatchNorm3d(c4)
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(c4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(1)


class DilatedResNet3D(nn.Module):
    """
    Dilated ResNet 3D architecture for RMSF prediction.
    
    This model uses residual blocks with dilated convolutions for better
    capturing multi-scale features in voxelized protein data.
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        channel_growth_rate: float = 1.5,
        num_residual_blocks: int = 4,
        dropout_rate: float = 0.3
    ):
        """
        Initialize DilatedResNet3D.
        
        Args:
            input_channels: Number of input channels (typically 4 or 5)
            base_filters: Number of filters in the first convolutional layer
            channel_growth_rate: Growth rate for channels in successive layers
            num_residual_blocks: Number of residual blocks
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv3d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        
        # Calculate channel sizes for each layer
        channels = [base_filters]
        for i in range(num_residual_blocks):
            channels.append(int(channels[-1] * channel_growth_rate))
        
        # Residual blocks with increasing dilation
        self.res_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            dilation = 2 ** (i % 3)  # Dilations: 1, 2, 4, 1, 2, 4, ...
            block = ResidualBlock3D(
                channels[i], channels[i+1], dilation=dilation, dropout_rate=dropout_rate
            )
            self.res_blocks.append(block)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(channels[-1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(1)


class MultipathRMSFNet(nn.Module):
    """
    Multi-path 3D CNN architecture for RMSF prediction.
    
    This model uses multiple parallel paths with different kernel sizes
    to capture features at different scales.
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        channel_growth_rate: float = 1.5,
        num_residual_blocks: int = 3,
        dropout_rate: float = 0.3
    ):
        """
        Initialize MultipathRMSFNet.
        
        Args:
            input_channels: Number of input channels (typically 4 or 5)
            base_filters: Number of filters in the first convolutional layer
            channel_growth_rate: Growth rate for channels in successive layers
            num_residual_blocks: Number of residual blocks in each path
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Calculate channel sizes
        c1 = base_filters
        c2 = int(c1 * channel_growth_rate)
        c3 = int(c2 * channel_growth_rate)
        
        # Initial convolution
        self.conv1 = nn.Conv3d(input_channels, c1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c1)
        
        # Multi-path branches with different kernel sizes
        self.path1 = self._create_path(c1, c2, kernel_size=3, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        self.path2 = self._create_path(c1, c2, kernel_size=5, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        self.path3 = self._create_path(c1, c2, kernel_size=7, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        
        # Fusion layer
        self.fusion = nn.Conv3d(c2 * 3, c3, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm3d(c3)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(c3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_path(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        blocks: int,
        dropout_rate: float
    ) -> nn.Sequential:
        """
        Create a path with multiple convolutional blocks.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            blocks: Number of blocks
            dropout_rate: Dropout rate
            
        Returns:
            Sequential container of blocks
        """
        layers = []
        
        # First block
        padding = kernel_size // 2
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        
        # Additional blocks
        for _ in range(blocks - 1):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout3d(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)
        
        # Multi-path processing
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        
        # Concatenate outputs
        out = torch.cat([out1, out2, out3], dim=1)
        
        # Fusion
        out = self.fusion(out)
        out = self.fusion_bn(out)
        out = F.relu(out)
        
        # Global average pooling
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        
        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out.squeeze(1)


def get_model(
    architecture: str,
    input_channels: int = 5,
    base_filters: int = 32,
    channel_growth_rate: float = 1.5,
    num_residual_blocks: int = 4,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Get a model based on the specified architecture.
    
    Args:
        architecture: Model architecture name
        input_channels: Number of input channels
        base_filters: Base number of filters
        channel_growth_rate: Channel growth rate
        num_residual_blocks: Number of residual blocks
        dropout_rate: Dropout rate
        
    Returns:
        PyTorch model
    """
    if architecture == "voxelflex_cnn":
        return VoxelFlexCNN(
            input_channels=input_channels,
            base_filters=base_filters,
            channel_growth_rate=channel_growth_rate,
            dropout_rate=dropout_rate
        )
    elif architecture == "dilated_resnet3d":
        return DilatedResNet3D(
            input_channels=input_channels,
            base_filters=base_filters,
            channel_growth_rate=channel_growth_rate,
            num_residual_blocks=num_residual_blocks,
            dropout_rate=dropout_rate
        )
    elif architecture == "multipath_rmsf_net":
        return MultipathRMSFNet(
            input_channels=input_channels,
            base_filters=base_filters,
            channel_growth_rate=channel_growth_rate,
            num_residual_blocks=num_residual_blocks,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")