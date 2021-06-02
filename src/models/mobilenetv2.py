from torchsummary import summary
from torchvision.models.mobilenetv2 import MobileNetV2

if __name__ == '__main__':
    model = MobileNetV2(width_mult=0.5)
    summary(model, (3, 160, 160), batch_size=1)
