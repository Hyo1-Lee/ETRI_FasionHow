import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        inputs = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = inputs + self.drop_path(x)
        return x
    
class ConvNeXtV2(nn.Module):
    def __init__(self, depths, dim, drop_rate):
        super(ConvNeXtV2, self).__init__()
        self.depths = depths
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=4, stride=4),
            LayerNorm(dim, eps=1e-6, data_format="channels_first")
        )
        self.block1 = Block(dim, drop_rate)
        
        self.down1 = nn.Sequential(
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, dim*2, kernel_size=2, stride=2)
        )
        self.block2 = Block(dim*2, drop_rate)
        
        self.down2 = nn.Sequential(
            LayerNorm(dim*2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*2, dim*4, kernel_size=2, stride=2)
        )
        self.block3 = Block(dim*4, drop_rate)
        
        self.down3 = nn.Sequential(
            LayerNorm(dim*4, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*4, dim*8, kernel_size=2, stride=2)
        )
        self.block4 = Block(dim*8, drop_rate)
        
        self.norm = nn.LayerNorm(dim*8, eps=1e-6)
        self.head = nn.Linear(dim*8, 18)
        
    def forward(self, inputs):
        x = self.stem(inputs)
        for _ in range(self.depths[0]):
            x = self.block1(x)
            
        x = self.down1(x)
        for _ in range(self.depths[1]):
            x = self.block2(x)
            
        x = self.down2(x)
        for _ in range(self.depths[2]):
            x = self.block3(x)
            
        x = self.down3(x)
        for _ in range(self.depths[3]):
            x = self.block4(x)
            
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x