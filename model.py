import torch
import torch.nn as nn


class KCnetwork(nn.Module):

    def __init__(self, dim_hidden, vocab_size, freq_words): #len(freq_words)=vocab_size, remove vocab_size?
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

        #unsqueeze should be removed if batched
        mu_hat = torch.argmax(torch.mm(data.unsqueeze(0), self.W), dim=1).squeeze()

        dW = torch.zeros(2*self.vocab_size, self.dim_hidden)

        data_p = data/self.p

        #Gather?
        dW[:,mu_hat] = epsilon*(data_p-self.W[:,mu_hat]*(torch.dot(data_p, self.W[:,mu_hat])))

        return dW

    def Step(self, data, epsilon):

        self.W += self.Grad(data, epsilon)

    def Hash(self, data, k):

        H = torch.zeros(len(data), self.dim_hidden)

        activations = torch.mm(data, self.W)

        _, index = torch.topk(activations, k=k, dim=1)

        H.scatter_(1, index, 1)

        return H



