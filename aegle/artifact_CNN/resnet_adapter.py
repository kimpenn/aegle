import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck, BasicBlock

class BottleneckAdapter(nn.Module):
    """
    A bottleneck adapter module: Conv1x1 (down) -> ReLU -> Conv1x1 (up).
    Placed in parallel to the main residual block convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int = None, bottleneck_dim: int = 64, stride: int = 1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
            
        # Apply stride in the down-projection to match the block's spatial downsampling
        self.down_proj = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, stride=stride, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up_proj = nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=False)
        
        # Initialize up_proj to near zero so the adapter starts as an identity function (roughly)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.act(self.down_proj(x)))

def inject_adapter(block: nn.Module, bottleneck_dim: int = 64):
    """
    Injects a BottleneckAdapter into a ResNet block (BasicBlock or Bottleneck).
    Wraps the forward method to add the adapter output to the residual path.
    """
    # Determine input and output channels
    # conv1 is always the first layer, so its in_channels is the block's input channels
    in_channels = block.conv1.in_channels
    
    # Determine output channels
    if isinstance(block, BasicBlock):
        out_channels = block.conv2.out_channels
    elif isinstance(block, Bottleneck):
        out_channels = block.conv3.out_channels
    else:
        return # Not a supported block type

    # Determine stride (default to 1 if not present, though ResNet blocks usually have it)
    stride = getattr(block, 'stride', 1)

    # Add the adapter module to the block
    block.adapter = BottleneckAdapter(in_channels, out_channels, bottleneck_dim, stride=stride)

    # Save the original forward method
    original_forward = block.forward

    # Define the new forward method
    def new_forward(x):
        identity = x

        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out)

        if isinstance(block, Bottleneck):
            out = block.relu(out)
            out = block.conv3(out)
            out = block.bn3(out)

        if block.downsample is not None:
            identity = block.downsample(x)

        # Add adapter output
        # Adapter takes 'x' (input to block) and produces output of shape 'out'
        adapter_out = block.adapter(x)
        
        out += adapter_out
        out += identity
        out = block.relu(out)

        return out

    # Bind the new forward method to the block instance
    block.forward = new_forward

class ResNetAdapter(nn.Module):
    """
    ResNet with PEFT Adapters.
    - Modifies first conv to accept N channels.
    - Injects adapters into residual blocks.
    - Freezes backbone, trains adapters + head + first conv.
    """
    def __init__(self, in_channels: int, num_classes: int, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        
        # Load backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            adapter_dim = 16 
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            adapter_dim = 64
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 1. Handle Input Channels
        if in_channels != 3:
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=old_conv.bias
            )
            
            # Initialize weights
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                mean_weight = torch.mean(old_conv.weight, dim=1, keepdim=True)
                for i in range(3, in_channels):
                    new_conv.weight[:, i:i+1, :, :] = mean_weight

            self.backbone.conv1 = new_conv

        # 2. Inject Adapters
        for name, module in self.backbone.named_modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                inject_adapter(module, bottleneck_dim=adapter_dim)

        # 3. Freeze Backbone
        for name, param in self.backbone.named_parameters():
            # Freeze everything by default
            param.requires_grad = False
            
            # Unfreeze adapters
            if "adapter" in name:
                param.requires_grad = True
            
            # Unfreeze final fc
            if "fc" in name:
                param.requires_grad = True
        
        # Explicitly unfreeze the first conv layer
        for param in self.backbone.conv1.parameters():
            param.requires_grad = True
                
        # Ensure the fc layer is correct for num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
