import torch
#import habana_frameworks.torch.hpu as torch_hpu
from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv = torch.nn.Conv2d(1, 4096, 3, 3)

    def forward(self,x):
        x = self.conv(x)
        out = torch.where(x)
        return out
device = torch.device('hpu')
net = MyNet().to(device)
net = detect_recompilation_auto_model(net)

device = torch.device('hpu')
#mask = torch.Tensor([[True,False,True],[False,True,True]]).to(device)
mask = torch.rand((1,1,4,4)).to(device)
#mask = expert_mask[1]
out = net(mask)
 
print(out)
net.analyse_dynamicity()
