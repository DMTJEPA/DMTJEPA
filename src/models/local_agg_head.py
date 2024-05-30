import math
from functools import partial
import torch
import torch.nn as nn
from src.utils.tensors import trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Class_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x, attention=False, mask=None, attn_type='cross-attn'):
        
        B, N, C = x.shape
        if attn_type == 'cross-attn':
            num_q = 1
            q = self.q(x[:,0]).unsqueeze(1).reshape(B, num_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        elif attn_type == 'self-attn':
            num_q = N   
            q = self.q(x).unsqueeze(1).reshape(B, num_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        if mask is not None:
            mask_temp = torch.cat([torch.ones(B,1).bool().cuda(), mask],dim=1).unsqueeze(1).unsqueeze(1).expand(-1,self.num_heads,-1,-1)
            attn = attn.masked_fill_(~mask_temp.bool(), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, num_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if attention:
            return x, attn
        else:
            return x


class LayerScale_Block_CA(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_cls=None, attention=False, mask=None, attn_type='cross-attn', loc=True):
        if x_cls is not None:
            u = torch.cat((x_cls,x),dim=1)
        else:
            u = x
            x_cls = x
        if attention:
            u_, cls_attn = self.attn(self.norm1(u), attn_type=attn_type, attention=True)
            return cls_attn            
        
        if loc:
            u_ = self.attn(self.norm1(u), mask=mask, attn_type=attn_type)
            if attn_type=='cross-attn':
                x = x_cls + self.drop_path(u_)                  
            elif attn_type=='self-attn':
                x = x_cls + self.drop_path(u_[:,:1])            
        else:
            if attn_type=='cross-attn':
                u_ = self.attn(self.norm1(u), mask=mask, attn_type=attn_type)
                x = x + self.drop_path(u_)                        
            elif attn_type=='self-attn':
                u_ = self.attn(self.norm1(x), mask=mask, attn_type=attn_type)
                x = x + self.drop_path(u_)                        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LocalAggHead(nn.Module):
    def __init__(self, in_dim, num_heads, k_num, k_size=3):
        super().__init__()
        self.cls_blocks = nn.ModuleList([
            LayerScale_Block_CA(
                dim=in_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU, Attention_block=Class_Attention,
                Mlp_block=Mlp)
            for i in range(2)])
        self.norm = partial(nn.LayerNorm, eps=1e-6)(in_dim)

        self.apply(self._init_weights)
        self.k_num = k_num
        self.k_size = k_size
        self.loc224_p14 = self.get_local_index(256, self.k_size)
        self.loc224 = self.get_local_index(196, self.k_size)
        self.loc96 = self.get_local_index(36, self.k_size)
        self.embed_dim = in_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, loc=False, attn_type='cross-attn'):
        cls_tokens = x.mean(dim=1, keepdim=True)
        if loc:
            k_size = self.k_size
            if x.shape[1] == 196:
                local_idx = self.loc224
            elif x.shape[1] == 36:
                if self.k_size == 14:
                    k_size = 6
                local_idx = self.loc96
            elif x.shape[1] == 256:
                local_idx = self.loc224_p14
            else:
                assert(False)

            x_norm = nn.functional.normalize(x, dim=-1)
            sim_matrix = x_norm[:,local_idx] @ x_norm.unsqueeze(2).transpose(-2,-1)
            top_idx = sim_matrix.squeeze().topk(k=self.k_num,dim=-1)[1].view(-1,self.k_num,1)

            x_loc = x[:,local_idx].view(-1,k_size**2-1,self.embed_dim)
            x_loc = torch.gather(x_loc, 1, top_idx.expand(-1, -1, self.embed_dim))
            for i, blk in enumerate(self.cls_blocks):
                if i == 0:
                    loc_tokens = blk(x_loc, cls_tokens.repeat(x.shape[1],1,1), attn_type=attn_type, loc=loc)
                else:
                    loc_tokens = blk(x_loc, loc_tokens, attn_type=attn_type, loc=loc)       
            loc_tokens = loc_tokens.view(x.shape)               
            x = self.norm(loc_tokens)
        else:
            for i, blk in enumerate(self.cls_blocks):
                if i == 0:
                    loc_tokens = blk(x, cls_tokens, attn_type=attn_type, loc=loc) 
                else:
                    cls_tokens = loc_tokens.mean(dim=1, keepdim=True)
                    loc_tokens = blk(loc_tokens, cls_tokens, attn_type=attn_type, loc=loc)                
            x = self.norm(loc_tokens)

        return x

    @staticmethod
    def get_local_index(N_patches, k_size):
        loc_weight = []
        w = torch.LongTensor(list(range(int(math.sqrt(N_patches)))))
        for i in range(N_patches):
            ix, iy = i//len(w), i%len(w)
            wx = torch.zeros(int(math.sqrt(N_patches)))
            wy = torch.zeros(int(math.sqrt(N_patches)))
            wx[ix]=1
            wy[iy]=1
            for j in range(1,int(k_size//2)+1):
                wx[(ix+j)%len(wx)]=1
                wx[(ix-j)%len(wx)]=1
                wy[(iy+j)%len(wy)]=1
                wy[(iy-j)%len(wy)]=1
            weight = (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
            weight[i] = 0
            loc_weight.append(weight.nonzero().squeeze())
        return torch.stack(loc_weight)