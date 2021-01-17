from model import *

data = torch.randn(30,2*20)
kc = KCnetwork(10,20,torch.randn(20))

print(kc.Energy(data))
print(kc.Grad(data[0], 0.1))
print(kc.Hash(data, 3))