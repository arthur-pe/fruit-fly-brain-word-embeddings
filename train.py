from recup_data_pick import Dataloader
from model import KCnetwork
import torch
from get_freq import get_freq

datas = ['/tempory/AMAL_DATASETS/save_massina_projet/vectors/vectorized_data_11_'+str(i)+'.csv' for i in range(10)]
#datas = ['/home/rito/PycharmProjects/ffbwe/useless/data-00.csv']

#freq, file_freq, = get_freq(datas)

#freq = torch.load('/home/rito/PycharmProjects/ffbwe/useless/tensor_freq.pt')
freq = torch.load('tensor_freq_100.pt')


model = KCnetwork(400, 20000, freq)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.W = torch.nn.Parameter(torch.load('weights_10.pt'))

model.to(device)

epochs = 15

loader = Dataloader(datas, 4000, shuffle=True, drop_last=True)

epsilon_zero = 5*10**-4
decay_rate = 10

for epoch in range(epochs):

    #print(model.W)

    epsilon = epsilon_zero*(1/(1+decay_rate*epoch))

    avg_l = 0
    n = 0

    for x in loader:
        n += 1
        #print(model.W)
        
        x = x.to(device)

        l = model.Energy(x)
       
        avg_l += l.item()
 
        if n%1000==0:
            print(epoch, n, l.item(), avg_l/n)

        model.Step(x, epsilon)

        #optim.zero_grad()
        #l.backward()
        #optim.step()

    avg_l /= n
    
    print()
    print(epoch, avg_l)

    torch.save(model.W, 'weights_'+str(len(datas))+'.pt')
    
