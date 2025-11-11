import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import nms
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
import math

class FeatureRecoveryModule(nn.Module):
    """Feature Recovery Module for visibility enhancement"""
    def __init__(self, in_channels=256):
        super(FeatureRecoveryModule, self).__init__()
        
        # Upsampling submodule
        self.upsample_conv = nn.Conv2d(in_channels, in_channels // 8, 1, 1, 0)
        
        # Multiscale Mapping submodule (Inception-like)
        self.mm_conv1 = nn.Conv2d(in_channels // 8, 4, 1, 1, 0)
        self.mm_conv3 = nn.Conv2d(in_channels // 8, 4, 3, 1, 1)
        self.mm_conv5 = nn.Conv2d(in_channels // 8, 4, 5, 1, 2)
        self.mm_conv7 = nn.Conv2d(in_channels // 8, 4, 7, 1, 3)
        
        # Fusion
        self.fusion = nn.Conv2d(16, 3, 3, 1, 1)
        
    def forward(self, x):
        # Upsampling
        x = self.upsample_conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Multiscale mapping
        f1 = self.mm_conv1(x)
        f3 = self.mm_conv3(x)
        f5 = self.mm_conv5(x)
        f7 = self.mm_conv7(x)
        
        # Concatenate
        G_x = torch.cat([f1, f3, f5, f7], dim=1)
        G_x = self.fusion(G_x)
        
        return G_x


class CommonBlockModule(nn.Module):
    """Common Block Module - shared between detection and restoration
    Uses first layers of ResNet-50 up to C2 (layer1)"""
    def __init__(self):
        super(CommonBlockModule, self).__init__()
        
        # Use ResNet-50 first blocks (up to C2)
        resnet50 = models.resnet50(pretrained=True)
        
        # Conv1 + BN + ReLU + MaxPool
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        
        # Layer1 (C2 in paper) - output channels = 256
        self.layer1 = resnet50.layer1
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # Output: fC2 with 256 channels
        
        return x


class RestorationSubnet(nn.Module):
    """Restoration Subnet for visibility enhancement"""
    def __init__(self, cb_module):
        super(RestorationSubnet, self).__init__()
        
        self.cb_module = cb_module
        self.fr_module = FeatureRecoveryModule(in_channels=256)
        
    def forward(self, foggy_image):
        # Extract features
        fC2 = self.cb_module(foggy_image)
        
        # Generate G(x) for atmospheric model
        G_x = self.fr_module(fC2)
        
        # Ensure G_x matches input size
        if G_x.shape[2:] != foggy_image.shape[2:]:
            G_x = F.interpolate(G_x, size=foggy_image.shape[2:], mode='bilinear', align_corners=False)
        
        # Image production: J(x) = G(x) * I(x) - G(x) + 1
        restored_image = G_x * foggy_image - G_x + 1
        
        return restored_image, fC2, G_x


class DetectionSubnet(nn.Module):
    """Detection Subnet using RetinaNet architecture"""
    def __init__(self, num_classes=3, cb_module=None):
        super(DetectionSubnet, self).__init__()

        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.models.detection.anchor_utils import AnchorGenerator

        # CRITICAL FIX: Use smaller anchors for better small object detection
        # Default RetinaNet anchors are too large for our objects
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [16, 32, 64, 128, 256])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        # Load WITHOUT pretrained weights for detection head
        # but WITH pretrained backbone
        self.retinanet = retinanet_resnet50_fpn(
            pretrained=False,  # Don't load pretrained detection head
            pretrained_backbone=True,  # But use pretrained ResNet-50
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            # CRITICAL: Lower IoU threshold to match more anchors for small objects
            fg_iou_thresh=0.4,  # Was 0.5 by default
            bg_iou_thresh=0.3,  # Was 0.4 by default
            # More detections
            detections_per_img=300,  # Was 300
            topk_candidates=2000,  # Was 1000
        )

        # CRITICAL FIX: Properly initialize classification head with focal loss bias
        # This ensures the network starts with reasonable classification predictions
        self._initialize_classification_head(num_classes)

        self.cb_module = cb_module

    def _initialize_classification_head(self, num_classes):
        """Initialize classification head with proper bias for focal loss"""
        # Prior probability of 0.05 for positive class (higher than default 0.01)
        # This helps prevent the network from collapsing to all-background predictions
        prior_prob = 0.05  # Increased from 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # Initialize classification head bias
        for module in self.retinanet.head.classification_head.cls_logits.modules():
            if isinstance(module, nn.Conv2d):
                # Initialize bias to focus on background initially
                # This prevents the "all predictions are background" problem
                nn.init.constant_(module.bias, bias_value)

    def forward(self, x, targets=None):
        if self.training and targets is not None:
            return self.retinanet(x, targets)
        else:
            self.retinanet.eval()
            return self.retinanet(x)


class DSNet(nn.Module):
    """
    DSNet: Dual-Subnet Network for Object Detection in Adverse Weather
    Paper: DSNet: Joint Semantic Learning for Object Detection in Inclement Weather Conditions
    Uses RetinaNet as Detection Subnet (as per original paper)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(DSNet, self).__init__()
        
        # Common Block Module (shared between both subnets)
        self.cb_module = CommonBlockModule()
        
        # Restoration Subnet
        self.restoration_subnet = RestorationSubnet(self.cb_module)
        
        # Detection Subnet (RetinaNet)
        self.detection_subnet = DetectionSubnet(num_classes=num_classes, cb_module=self.cb_module)
        
        self.num_classes = num_classes
        
    def forward(self, images, targets=None):
        """
        Args:
            images: Input foggy images [B, 3, H, W]
            targets: List of dictionaries containing 'boxes' and 'labels' (training only)
        
        Returns:
            If training: loss_dict and restored images
            If inference: detections and restored images
        """
        if self.training:
            # During training: use restoration subnet + detection
            restored_images, fC2, G_x = self.restoration_subnet(images)
            
            # Detection on restored images
            loss_dict = self.detection_subnet(restored_images, targets)
            
            return loss_dict, restored_images, G_x
        else:
            # During inference: 
            # Option 1: Direct detection on foggy images (as per paper - restoration subnet not used in inference)
            detections = self.detection_subnet(images)
            
            # Option 2: Can also generate restored images for visualization
            with torch.no_grad():
                restored_images, _, _ = self.restoration_subnet(images)
            
            return detections, restored_images


class FocalLoss(nn.Module):
    """Focal Loss as described in RetinaNet paper"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_dsnet_model(num_classes=2, pretrained=True):
    """
    Factory function to create DSNet model
    
    Args:
        num_classes: Number of object classes (default: 2 for person and car)
        pretrained: Whether to use pretrained ResNet-50 backbone
    
    Returns:
        DSNet model
    """
    model = DSNet(num_classes=num_classes, pretrained=pretrained)
    return model


# Utility functions for training and inference
def dsnet_collate_fn(batch):
    """Custom collate function for DSNet DataLoader"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


def prepare_targets_for_dsnet(annotations, image_id):
    """
    Convert COCO-style annotations to DSNet format
    
    Args:
        annotations: List of annotation dicts with 'bbox' and 'category_id'
        image_id: Image identifier
    
    Returns:
        Dictionary with 'boxes' and 'labels' tensors
    """
    boxes = []
    labels = []
    
    for ann in annotations:
        # COCO format: [x, y, width, height]
        x, y, w, h = ann['bbox']
        # Convert to [x1, y1, x2, y2]
        boxes.append([x, y, x + w, y + h])
        labels.append(ann['category_id'])
    
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64)
    }
