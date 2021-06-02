from torchsummary import summary
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

if __name__ == '__main__':
    model = mobilenet_v3_small()
    summary(model, (3, 160, 160), batch_size=1)
