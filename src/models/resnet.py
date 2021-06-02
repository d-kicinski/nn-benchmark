from torchsummary import summary
from torchvision.models.resnet import resnet18, resnet34, resnet50

if __name__ == '__main__':
    model = resnet18()
    summary(model, (3, 160, 160), batch_size=1)
