from nlp_mnist import NlpMnist
# from tf.model import Transformer
from torch.utils.data import DataLoader
from torch import nn 
import torch.optim

batch_size = 32
nc = 512
lr_init = 0.05
depth = 8
# train_dataset = NlpMnist(seq_len=1024, num_varibles=(20, 500))
train_dataset = NlpMnist()
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

device = "cuda:0"


class MaxStateSuper(torch.nn.Module):
    def __init__(self, dim_size, heads):
        super(MaxStateSuper, self).__init__()
        self.heads = heads
        assert dim_size % heads == 0, "Dimension size must be divisible by head size."
        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha3 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha4 = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None):
        b, s, d = x.shape
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)
        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)
        if  state is None:

            out4, _ = torch.cummax(out2, dim=2)
            state = out4[:, :, -1:]
        else:
            out4,_ = torch.cummax(torch.cat([state,out2], dim=2), dim=2)
            state = out4[:, :, -1:]
            out4 = out4[:, :, -1:]
        out = self.gen_model(out, out1, out2, out3, out4)
        # print("out", out.shape, x.shape, out1.shape)
        # exit()

        out = out.transpose(1, 2).contiguous().view(b, s, d)

        return out, state

    def gen_model(self, a, b, c, d, e):
        term1 = a * b
        term2 = self.alpha1 * b + self.alpha2 * d
        term3 = a * (self.alpha3 * e + d)
        term4 = b * (c + e)
        return term1 + term2 + term3 + term4 + c * e

    # def gen_model(self, a, b, c, d, e):
    #     x = self.alpha1 * b + self.alpha2*d +a
    #     x = a*b + x
    #     x = self.alpha3 *a*e+x
    #     x = b*c +x
    #     x= b*e  +x
    #     x= c*e +x
    #     x= a*d+x
    #
    #
    #     # ab = a * b
    #     # ad = a * d
    #     # ae = a * e
    #     # bc = b * c
    #     # be = b * e
    #     # ce = c * e
    #
    #     # # # 初始计算
    #
    #     # x = self.layer_norm(ab + 2 * b )+ 2 * ae + 2 * d + bc + be + a + self.layer_norm(ce + ad)
    #
    #     return x
    #


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size):
        super(FeedForward, self).__init__()
        self.ffn1 = torch.nn.Linear(hidden_size, hidden_size)
        self.ffn2 = torch.nn.Linear(hidden_size, hidden_size)
        self.gate = torch.nn.Linear(hidden_size, hidden_size)

        self.relu = torch.nn.ReLU()
        # self.gr = torch.nn.Dropout(0.02)

    def forward(self, x):
        x1 = self.ffn1(x)
        x2 = self.relu(self.gate(x))
        xx = x1 * x2
        x = self.ffn2(xx)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = MaxStateSuper(hidden_size, num_heads)
        # self.self_attention = MaxState(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None, ):
        x1, state = self.self_attention(x, state)
        x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)

        return x, state


class SamOut(torch.nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers):
        super(SamOut, self).__init__()
        self.em = torch.nn.Embedding(voc_size, hidden_size, padding_idx=0)

        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(hidden_size, num_heads) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, 10, bias=False)
        # self.alpha = [torch.nn.Parameter(torch.tensor(0.5)) for i in range(num_layers)]
        # self.layer_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, x, state=None):
        x = self.em(x)

        if state is None:
            state = [None] * len(self.decoder_layers)
        i = 0

        for ii, decoder_layer in enumerate(self.decoder_layers):
            x1, state[i] = decoder_layer(x, state[i])
            x = x1 + x
            i += 1

        
        x = self.head(x)

        x = x[:, -1, :]
        # x = x.mean(dim=1)
        return x
    

net = SamOut(hidden_size=nc, num_heads=32, num_layers=depth, voc_size=len(train_dataset.char2idx)).to(device)
my_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr_init, weight_decay=1e-5)
def lambda1(epoch):
    warmup = 400
    if epoch < warmup:
        mul = (warmup - epoch) / warmup * (0.1 - 1) + 1 
    elif epoch < 4000:
        mul = 1
    else:
        mul = 0.2
        
    return mul
    
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)     

# print(net)
net.train()
for epoch in range(5000):
    train_loss = 0
    lr = lr_scheduler.get_last_lr()[0]
    for ain, aout in train_loader:
        # print(ain.shape, aout.shape)
        ain = ain.to(device)
        aout = aout.to(device).view(-1)
        optimizer.zero_grad()
        y = net(ain)
        # print(y.shape)
        # exit()

        loss = my_loss(y, aout)
        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        # net.encoder.renew(lr)
    lr_scheduler.step()

    print(epoch+1, f"train_loss {train_loss:.5f}, lr {lr:.3f}")
    # break

net.eval()

total = 0
err = 0
with torch.no_grad():
    for ain, aout in train_dataset:
        ain = ain.to(device)
        ain.unsqueeze_(0)
        aout = aout.item()
        pred = net(ain)
        # print(pred.shape)
        pred = torch.argmax(pred, -1).item()

        total += 1
        
        if pred != aout: 
            for idx in ain[0]:
                idx = int(idx)
                c = train_dataset.alphabet[idx]
                if c == train_dataset.BLANK: continue
                print(train_dataset.alphabet[idx], end='')
            print(f" {pred} != {aout}(expected)")
            err += 1
        if total >= 1000: break

print(f"error {err}/{total}")
