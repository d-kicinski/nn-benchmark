from torchvision.models import mnasnet0_5, mnasnet1_0
from torchsummary import summary

if __name__ == '__main__':
    model = mnasnet0_5()
    summary(model, (3, 160, 160), batch_size=1, device="cpu")
