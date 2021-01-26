import torch
import torch.nn as nn

class KCnetwork(nn.Module):

    def __init__(self, dim_hidden, vocab_size, freq_words):
        super(KCnetwork, self).__init__()

        self.W = nn.Parameter(torch.randn(2*vocab_size, dim_hidden))
        self.p = torch.cat((freq_words, freq_words))

        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size

    def Energy(self, data):

        mu_hat = torch.argmax(torch.mm(data, self.W), dim=1)

        E = -torch.sum(torch.sum(data/self.p.unsqueeze(0)*self.W[:,mu_hat].transpose(0,1), dim=1)
                       /torch.sqrt(torch.sum((self.W[:,mu_hat]*self.W[:,mu_hat]).transpose(0,1), dim=1)))

        return E

    def Grad(self, data, epsilon):

        mu_hat = torch.argmax(torch.mm(data, self.W), dim=1)

        dW = torch.zeros(self.dim_hidden,2*self.vocab_size)

        data_p = data/self.p

        src = epsilon*(data_p-self.W[:,mu_hat].transpose(0,1)*(torch.sum(data_p*self.W[:,mu_hat].transpose(0,1), dim=1)).unsqueeze(1))

        dW.scatter_add_(dim=0,index=mu_hat.unsqueeze(1),src=src)

        dW = dW.transpose(0,1)

        return dW

    def Step(self, data, epsilon):

        with torch.no_grad():

            self.W += self.Grad(data, epsilon)

    def Hash(self, data, k):

        H = torch.zeros(len(data), self.dim_hidden)

        activations = torch.mm(data, self.W)

        _, index = torch.topk(activations, k=k, dim=1)

        H.scatter_(1, index, 1)

        return H



