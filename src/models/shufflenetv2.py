from torchsummary import summary
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0

if __name__ == '__main__':
    model = shufflenet_v2_x1_0()
    summary(model, (3, 160, 160), batch_size=1)
