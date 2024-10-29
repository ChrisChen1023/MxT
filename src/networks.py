import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mamba_ssm import Mamba
import time
import math

from pdb import set_trace as stx
import numbers
from einops import rearrange

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)     


        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=0, groups=dim, dilation=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=0, groups=dim, dilation=2,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, x, spatial):
        
        
        x = self.project_in(x)
        
        
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        
        x_c = self.pool(x)

        x_bc = x_c.mean(dim=1, keepdim=True)
        x = x+ x_bc

        return x

##########################################################################
## SRSA
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.pool = nn.AdaptiveAvgPool2d(8)
        
        
        #### SA attention
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)
        ##########
        self.lce = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        #### Spatial Attention
        qkv2 = self.qkv_dwconv2(self.qkv2(x))
        q2, k2, v2 = qkv2.chunk(3, dim=1)
        
        v_lce = v2

        v2 = self.pool(v2)
        k2 = self.pool(k2)

        b2, c2, h2, w2 = v2.shape



        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k2 = rearrange(k2, 'b (head c) h2 w2 -> b head c (h2 w2)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h2 w2 -> b head c (h2 w2)', head=self.num_heads)

        v2 = torch.nn.functional.normalize(v2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)


        attn_s = (q2.transpose(-2, -1) @ k2) * self.temperature2
        attn_s = attn_s.softmax(dim=-1)

        out2 = (v2 @ attn_s.transpose(-2, -1))


        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.lce(v_lce) + out2
        #######################


        out = self.project_out(out2)
        return out


##########################################################################
### Hybrid Module
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        ##### Try Mamba
        self.mam = MambaLayer(dim)
        self.attn = Attention(dim, num_heads, bias)
        #####

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.fuse = nn.Conv2d(dim + dim, dim, kernel_size=1, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x, pos, order = x[0], x[1], x[2]
        x_spatial = x

        x_m = self.mam(self.norm1(x), pos, order)
        x_sa = self.attn(self.norm3(x))
        x = x + self.fuse(torch.cat((x_m,x_sa),dim=1))
        
        x = x + self.ffn(self.norm2(x), x_spatial)

        return {0:x, 1:pos, 2:order}
        




##########################################################################
## Overlapped image patch embedding with 3x3 GatedConv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.gproj1 = nn.Conv2d(in_c, embed_dim*2, kernel_size=3,stride=1,padding=1,bias=bias)


    def forward(self, x):
        #x = self.proj(x)
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)

        x = F.gelu(x1) * x2

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
        
        self.proj = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=1, padding=1, groups=n_feat * 2, bias=False)
    def forward(self, x, mask):

        out = self.body(x)
        return out



class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
	
    def forward(self, x, mask):
        return self.body(x)


##########################################################################
##---------- MxT -----------------------
class MxT(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):

        super(MxT, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])


        ###########################

        self.output_before = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Sequential(nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                                    )        


    def forward(self, inp_img, mask_whole, mask_half, mask_quarter,mask_tiny, pos1, pos2, pos3, pos4, pos1_dec,order256,order128,order64,order32):
        
        inp_enc_level1 = self.patch_embed(torch.cat((inp_img,mask_whole),dim=1))

        out_enc_level1 = self.encoder_level1({0:inp_enc_level1, 1:pos1, 2:order256})

        inp_enc_level2 = self.down1_2(out_enc_level1[0],mask_whole)
        out_enc_level2 = self.encoder_level2({0:inp_enc_level2, 1:pos2, 2:order128})

        inp_enc_level3 = self.down2_3(out_enc_level2[0],mask_half)
        out_enc_level3 = self.encoder_level3({0:inp_enc_level3, 1:pos3, 2:order64})

        inp_enc_level4 = self.down3_4(out_enc_level3[0],mask_quarter)

        latent = self.latent({0:inp_enc_level4, 1:pos4, 2:order32})

        inp_dec_level3 = self.up4_3(latent[0],mask_tiny)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3[0]], 1)

        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3({0:inp_dec_level3, 1:pos3, 2:order64})

        inp_dec_level2 = self.up3_2(out_dec_level3[0],mask_quarter)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2[0]], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2({0:inp_dec_level2, 1:pos2, 2:order128})

        inp_dec_level1 = self.up2_1(out_dec_level2[0],mask_half)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1[0]], 1)

        out_dec_level1 = self.decoder_level1({0:inp_dec_level1, 1:pos1_dec, 2:order256})


        out_dec_level1 = self.output(out_dec_level1[0])

        out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2
        return out_dec_level1
        

        
        
        
        
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
        self.mamba2 = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True)
        
    
    @autocast(enabled=False)
    def forward(self, x, pe, order):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]

        reversed_x1 = x.clone()
        
        r_order = order.clone()

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        
        
        
        x1_flat = reversed_x1.reshape(B, C, n_tokens).transpose(-1, -2)
        x1_flat = x1_flat + pe[:n_tokens, :]


        x1_norm = self.norm(x1_flat)

        x1_mamba = self.mamba(x1_norm)
        


        out1 = x1_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        
        out = out1 

        return out
        
        
        

