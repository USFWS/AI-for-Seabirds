import torchvision.models

## Swin models
#weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
#model = torchvision.models.swin_t(weights= weights)

# resent
torchvision.models.resnet50(weights="IMAGENET1K_V1")