from model import *

def train_autograd():
    torch.manual_seed(1)
    kc = KCnetwork(8,5,torch.tensor([0.2,0.2,0.2,0.2,0.2]))

    dic = {'a' : torch.tensor([0,0,0,0,1]), 'b' : torch.tensor([0,0,0,1,0]), 'c' : torch.tensor([0,0,1,0,0]), 'd' : torch.tensor([0,1,0,0,0]), 'e' : torch.tensor([1,0,0,0,0])}

    w1 = dic['a']+dic['a']
    w2 = dic['b']+dic['a']

    w3 = dic['c']+dic['c']
    w4 = dic['d']+dic['c']

    u1 = dic['c']
    u2 = dic['a']

    x1 = torch.cat((w1, u1)).float()
    x2 = torch.cat((w2, u1)).float()

    y1 = torch.cat((w3, u2)).float()
    y2 = torch.cat((w4, u2)).float()

    s = torch.stack((x1,x2,y1,y2))

    optim = torch.optim.Adam(kc.parameters(), lr=0.001)

    to_hash = torch.cat((torch.zeros(5),dic['c'])).unsqueeze(0)
    to_hash2 = torch.cat((torch.zeros(5),dic['a'])).unsqueeze(0)

    print(kc.Energy(s))

    for i in range(1000):

        l = kc.Energy(s)
        optim.zero_grad()
        l.backward()
        optim.step()


    print(kc.Energy(s))

    print(kc.Hash(x1.unsqueeze(0), 3))
    print(kc.Hash(to_hash, 3))
    print(kc.Hash(to_hash2, 3))

def train_manualgrad():
    torch.manual_seed(1)
    kc = KCnetwork(8, 5, torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))

    dic = {'a': torch.tensor([0, 0, 0, 0, 1]), 'b': torch.tensor([0, 0, 0, 1, 0]), 'c': torch.tensor([0, 0, 1, 0, 0]),
           'd': torch.tensor([0, 1, 0, 0, 0]), 'e': torch.tensor([1, 0, 0, 0, 0])}

    w1 = dic['a'] + dic['a']
    w2 = dic['b'] + dic['a']

    w3 = dic['c'] + dic['c']
    w4 = dic['d'] + dic['c']

    u1 = dic['c']
    u2 = dic['a']

    x1 = torch.cat((w1, u1)).float()
    x2 = torch.cat((w2, u1)).float()

    y1 = torch.cat((w3, u2)).float()
    y2 = torch.cat((w4, u2)).float()

    s = torch.stack((x1, x2, y1, y2))

    to_hash = torch.cat((torch.zeros(5), dic['c'])).unsqueeze(0)
    to_hash2 = torch.cat((torch.zeros(5), dic['a'])).unsqueeze(0)

    print(kc.Energy(s))

    for i in range(1000):
        kc.Step(s, 0.001)

    print(kc.Energy(s))

    print(kc.Hash(x1.unsqueeze(0), 3))
    print(kc.Hash(to_hash, 3))
    print(kc.Hash(to_hash2, 3))
