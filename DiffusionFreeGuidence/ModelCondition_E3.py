
   
import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import quantization as qt
from einops import rearrange
import numpy as np
def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb=None, cemb=None):
        x = self.c1(x) + self.c2(x)
        return x


class DownSample1d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=(2, 1), padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=(2, 1), padding=2)

    def forward(self, x, temb=None, cemb=None):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb=None, cemb=None):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x
    

class UpSample1d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, (2, 1), (2,2), (1,0))

    def forward(self, x, temb=None, cemb=None):
        x = x.unsqueeze(1)
        x = self.t(x)
        x = self.c(x)
        return x

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)
    
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h



class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim // 2, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()


    def forward(self, x):
        h = self.block1(x) # torch.Size([4, 64, 276, 128]) B, C, T, D
        # h += self.temb_proj(temb)[:, :, None, None]
        # h += self.cond_proj(labels).transpose(1,2).contiguous()[:, :, :, None]
        # h += cemb
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h) # torch.Size([2, 64, 500, 128])
        return h

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
        
class UNet(nn.Module):
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
        self.head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        self.cemb_downblocks = nn.ModuleList()
        self.cemb_upblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                self.cemb_downblocks.append(DownSample1d(1))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
                self.cemb_upblocks.append(UpSample1d(1))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )

        self.preconv_prosody = torch.nn.Sequential(
            nn.Conv1d(1024, tdim // 2, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(tdim // 2),
            Mish())
        self.preconv_timbre = torch.nn.Sequential(
            nn.Conv1d(512, tdim // 2, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(tdim // 2),
            Mish())
        self.preconv_content = torch.nn.Sequential(
            nn.Conv1d(1024, tdim // 2, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(tdim // 2),
            Mish())
        
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        sample_rate = 24_000
        channels = 1
        ratios = [8, 5, 4, 2]
        ratios = list(reversed(ratios))
        hop_length = np.prod(ratios)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / hop_length) * 10))  # = 32
        self.quantizer = qt.ResidualVectorQuantizer(
            dimension=128,
            n_q=n_q,
            bins=1024,
        )

    def forward(self, labels):
        # Timestep embedding
        # temb = self.time_embedding(t) # torch.Size([8, 512])

        # cond embedding # torch.Size([8, 380, 1024])
        pro_emb = self.preconv_prosody(labels['pro'].permute(0, 2, 1).contiguous())
        tim_emb = self.preconv_timbre(labels['tim'].permute(0, 2, 1).contiguous())
        con_emb = self.preconv_content(labels['con'].permute(0,2,1).contiguous())
        cemb = (pro_emb + tim_emb + con_emb) # torch.Size([2, 200, 256])
        # print(cemb)
        # print('cemb:', cemb.max(), cemb.min())
        # Downsampling
        x = cemb
        h = self.head(x.unsqueeze(1)) # torch.Size([8, 128, 400, 128])
        hs = [h]
        
        i = 0
        for layer in self.downblocks:
            h = layer(h)
            # print(h.shape)
            # if isinstance(layer, DownSample):
            #     cemb = self.cemb_downblocks[i](cemb.unsqueeze(1), h, temb).squeeze(1)
            #     i += 1
            hs.append(h)
        # print([h.shape for h in hs])
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Upsampling
        i = 0
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h)
            # print(h.shape)
            # if isinstance(layer, UpSample):
            #     cemb = self.cemb_upblocks[i](cemb, h, temb).squeeze(1)
            #     i += 1
        h = self.tail(h).squeeze(1).transpose(1,2).contiguous()
        # print(h)
        # print('h:', h.max(), h.min())
        assert len(hs) == 0


        # quant h torch.Size([1, 128, 137])
        qv = self.quantizer.forward(h, 75, 6.0)
        # wav (padding) [1,2,3,-1,-1,0,0] -> [110,110, ] -> [1,2,3,0,0,0,0]
        # codes = self.quantizer.encode(h, 75, 6.0)
        # codes = codes.transpose(0, 1)
        return qv.quantized, qv.penalty, qv.codes.permute(1,2,0), h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size])
    # resB = ResBlock(128, 256, 64, 0.1)
    # x = torch.randn(batch_size, 128, 32, 32)
    # t = torch.randn(batch_size, 64)
    # labels = torch.randn(batch_size, 64)
    # y = resB(x, t, labels)
    y = model(x, t, labels)
    print(y.shape)

