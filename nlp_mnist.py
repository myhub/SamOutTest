import torch
import os
# import random
from random import SystemRandom
random = SystemRandom()
from torch.utils.data import Dataset
import sys
from io import StringIO

class NlpMnist(Dataset): 
    def __init__(self, seq_len=64, num_varibles=(2, 8)):
        self.a_z = []
        self.digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for i in range(26):
            self.a_z += chr(i + ord('a'))

        self.BLANK = chr(2)
        self.alphabet = [self.BLANK] + self.digits + self.a_z + [' ', ';', '=', '(', ')']

        assert len(num_varibles) == 2
        assert num_varibles[0] <= num_varibles[1]

        self.seq_len = seq_len
        self.num_varibles = num_varibles

        self.names = []
        for e1 in self.a_z:
            self.names.append(e1)
            for e2 in range(100):
                self.names.append(f"{e1}{e2}")
        
        assert num_varibles[1] <= len(self.names), f"{len(self.names)}"

        self.char2idx = {}
        for i, c in enumerate(self.alphabet):
            self.char2idx[c] = i

    def __len__(self):
        return 1024
    
    def __getitem__(self, index):
        _ = index
        sel = random.sample(self.names, k=random.randint(self.num_varibles[0], self.num_varibles[1]))
 
        names = []
        ain = ""
        for k in sel:
            ain += k
            ain += '=' + str(random.randint(0, 9))
            ain += ";"
            names.append(k)
            if len(ain) > self.seq_len - 20: 
                break

        stdout_new = StringIO()
        stdout_save = sys.stdout

        loc = {}
        name = random.choice(names)
        ain += "print(" + name + ') '
        sys.stdout = stdout_new
        exec(ain, None, loc)
        sys.stdout = stdout_save

        aout = stdout_new.getvalue().strip()

        assert len(ain) <= self.seq_len

        # ain =  self.BLANK * (self.seq_len - len(ain) - 2)  + ain +self.BLANK*2
        
        aout = [ord(aout[0]) - ord('0')]
        ain = [self.char2idx[_] for _ in ain]

        return torch.LongTensor(ain), torch.LongTensor(aout)


if __name__ == '__main__':
    ds = NlpMnist(seq_len=128, num_varibles=(20, 500))

    for i in range(15000):
        ain, aout = ds[i]
        print(' | ', end="")
        for idx in ain:
            c = ds.alphabet[idx]
            if c == ds.BLANK: continue
            print(ds.alphabet[idx], end='')
        print(' | ', end="")
        for idx in aout:
            # c = ds.alphabet[idx]
            # if c == ds.BLANK: continue
            print(idx.item(), end='')
        print(' | ')
