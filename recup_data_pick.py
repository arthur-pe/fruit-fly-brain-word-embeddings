from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import pickle
import torch
import random

class Dataloader:

    def __init__(self, datas, batch_size, shuffle=True, drop_last=True):

        self.datas = datas
        if shuffle == True:
            random.shuffle(self.datas)

        self.n_file = 0
        self.perms = [[] for i in range(len(self.datas))]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.next_file()

    def to_dense(self, x, length=40000):

        try:
            vect = torch.zeros(length)
            ones = x.split('\n')
            for one in ones:
                pos = one.split('\t')[0]
                position = re.findall(r'\(0,(.*)\)', pos, re.DOTALL)
                position = int(position[0])
                vect[position] = 1.0

            return vect
        except:

            vect = torch.zeros(length)

            return vect

    def next_file(self):
        self.current_file = pd.read_csv(self.datas[self.n_file], header=None)
        self.current_file_len = len(self.current_file[0])
        self.n = 0
        if len(self.perms[self.n_file]) == 0:
            self.rand_perm = np.random.permutation(self.current_file_len)
            self.perms[self.n_file] = self.rand_perm
        else:
            self.rand_perm = self.perms[self.n_file]

        self.n_file += 1

    def init_files(self):

        self.n_file = 0
        self.next_file()

    def __iter__(self):
        self.init_files()

        return self

    def __next__(self):

        if self.drop_last == True:

            if self.n+self.batch_size >= self.current_file_len:

                if self.n_file >= len(self.datas):
                    raise StopIteration

                else:
                    self.next_file()
        else:

            if self.n >= self.current_file_len:

                if self.n_file >= len(self.datas):
                    raise StopIteration

                else:
                    self.next_file()

        batch = []
        batch_n = 0

        if self.shuffle==True:
            while self.n < self.current_file_len and batch_n < self.batch_size:
                batch.append(self.to_dense(self.current_file[0][self.rand_perm[self.n]]))

                self.n += 1
                batch_n += 1
        else:
            while self.n < self.current_file_len and batch_n < self.batch_size:
                batch.append(self.to_dense(self.current_file[0][self.n]))

                self.n += 1
                batch_n += 1

        batch = torch.stack(batch)

        return batch


