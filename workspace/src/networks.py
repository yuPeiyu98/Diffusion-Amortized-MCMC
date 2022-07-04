import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

###################################################################
####################### BASIC ARCH COMPONENTS #####################
###################################################################

##### + LEGACY

class GLU(nn.Module):
    """ (LEGACY) GLU activation halves the channel number once applied """
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

##### + ACTIVE

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class UpBlock(nn.Module):
    """ Upsample the feature map by a factor of 2x """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01, 
        use_spc_norm=False
    ):
        super(UpBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=2, 
                mode='bilinear'
            ),
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1,
                    padding=1, 
                    bias=False
                ),
                mode=use_spc_norm
            ),              
            nn.InstanceNorm2d(
                out_channels,
                affine=True, 
                track_running_stats=False
            ),
            nn.LeakyReLU(
                r, 
                inplace=True
            )
        )
    
    def forward(self, x):
        return self.block(x)

class SameBlock(nn.Module):
    """ shape-preserving feature transformation """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(SameBlock, self).__init__()
        
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False 
                ),
                mode=use_spc_norm
            ),                
            nn.InstanceNorm2d(
                out_channels,
                affine=True, 
                track_running_stats=False
            ),
            nn.LeakyReLU(
                r, 
                inplace=True
            )
        )
    
    def forward(self, x):
        return self.block(x)

class SameBlockWOLReLU(nn.Module):
    """ shape-preserving feature transformation """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(SameBlockWOLReLU, self).__init__()
        
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False 
                ),
                mode=use_spc_norm
            ),                
            nn.InstanceNorm2d(
                out_channels,
                affine=True, 
                track_running_stats=False
            )
        )
    
    def forward(self, x):
        return self.block(x)

class SameBlockPreLReLU(nn.Module):
    """ shape-preserving feature transformation """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(SameBlockPreLReLU, self).__init__()
        
        self.block = nn.Sequential(
            nn.LeakyReLU(
                r, 
                inplace=True
            ),
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False 
                ),
                mode=use_spc_norm
            ),                
            nn.InstanceNorm2d(
                out_channels,
                affine=True, 
                track_running_stats=False
            )            
        )
    
    def forward(self, x):
        return self.block(x)

class DownBlock2x(nn.Module):
    """ down-sample the feature map by a factor of 2x """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(DownBlock2x, self).__init__()
        
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=4, 
                    stride=2,
                    padding=1, 
                    bias=False
                ),
                mode=use_spc_norm
            ),
            nn.InstanceNorm2d(
                out_channels,
                affine=True, 
                track_running_stats=False
            ),
            nn.LeakyReLU(
                r, 
                inplace=True
            )
        )
    
    def forward(self, x):
        return self.block(x)

class DownBlock4x(nn.Module):
    """ down-sample the feature map by a factor of 4x """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(DownBlock4x, self).__init__()
        
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=8, 
                    stride=4,
                    padding=2, 
                    bias=False
                ), 
                mode=use_spc_norm
            ),
            nn.InstanceNorm2d(
                out_channels,
                affine=True, 
                track_running_stats=False
            ),
            nn.LeakyReLU(
                r, 
                inplace=True
            )
        )
    
    def forward(self, x):
        return self.block(x)

##### + W/O Normalization

class DownBlock2xWONorm(nn.Module):
    """ down-sample the feature map by a factor of 2x """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(DownBlock2xWONorm, self).__init__()
        
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=4, 
                    stride=2,
                    padding=1, 
                    bias=True
                ),
                mode=use_spc_norm
            ),            
            nn.LeakyReLU(
                r, 
                inplace=True
            )
        )
    
    def forward(self, x):
        return self.block(x)

class DownBlock4xWONorm(nn.Module):
    """ down-sample the feature map by a factor of 2x """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(DownBlock4xWONorm, self).__init__()
        
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=8, 
                    stride=4,
                    padding=2, 
                    bias=True
                ),
                mode=use_spc_norm
            ),            
            nn.LeakyReLU(
                r, 
                inplace=True
            )
        )
    
    def forward(self, x):
        return self.block(x)

##### + FUTURE

class ResBlock(nn.Module):
    """ shape-preserving resnet-like feature transformation """
    def __init__(
        self,
        in_channels, 
        out_channels, 
        r=.01,
        use_spc_norm=False
    ):
        super(ResBlock, self).__init__()
        self.blocks = nn.ModuleList([
            # input projection
            nn.Sequential(
                spectral_norm(
                    nn.Conv2d(
                        in_channels, 
                        in_channels, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1, 
                        bias=False 
                    ),
                    mode=use_spc_norm
                ),                
                nn.InstanceNorm2d(
                    in_channels,
                    affine=True, 
                    track_running_stats=False
                )
            ),            

            # output projection
            nn.Sequential(
                nn.LeakyReLU(
                    r, 
                    inplace=True
                ),
                SameBlock(
                    in_channels, 
                    out_channels, 
                    r=r,
                    use_spc_norm=use_spc_norm
                )
            )            
        ])                    

    def forward(self, x):
        return self.blocks[1](self.blocks[0](x) + x)

###################################################################
###################### BASIC NET COMPONENT ########################
###################################################################

class BaseNetwork(nn.Module):        
    def __init__(self, module_name):
        super(BaseNetwork, self).__init__()
        self.module_name = module_name

    def init_weights(self, init_type='orthogonal', gain=1.):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        blob/9451e70673400885567d08a9e97ade2524c700d0/models/
        networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and \
                (classname.find('Conv') != -1 \
                or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(
                        m.weight.data, 
                        0.0, 
                        gain
                    )
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(
                        m.weight.data, 
                        gain=gain
                    )
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(
                        m.weight.data, 
                        a=0, 
                        mode='fan_in'
                    )
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(
                        m.weight.data, 
                        gain=gain
                    )

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

###################################################################
########################### VAE NETWORKS ##########################
###################################################################

class Encoder8(BaseNetwork):
    """
    Bottom-up 8x8 encoder architecture
    """
    def __init__(
        self, 
        module_name='Encoder8',
        block_chns=[3, 128, 64, 16, 1],
        r=.01,
        use_spc_norm=False,
        init_weights=True
    ):
        super(Encoder8, self).__init__(module_name=module_name)

        ##### + encoding blocks
        self.e_blocks = [
            SameBlockWOLReLU(
                block_chns[0], 
                block_chns[1], 
                r=r, 
                use_spc_norm=use_spc_norm
            )
        ]

        for i in range(1, len(block_chns) - 1, 1):
            self.e_blocks.append(
                SameBlockPreLReLU(
                    block_chns[i], 
                    block_chns[i + 1], 
                    r=r,
                    use_spc_norm=use_spc_norm
                )
            )

        self.e_blocks = nn.ModuleList(self.e_blocks)        

        self.m_head = NonLinearHead(
            in_channels=64, 
            out_channels=64,       
            use_bias=False,
            r=.01,
            use_spc_norm=False,
            init_weights=True
        )
        self.v_head = NonLinearHead(
            in_channels=64, 
            out_channels=64,       
            use_bias=False,
            r=.01,
            use_spc_norm=False,
            init_weights=True
        )

        if init_weights:
            self.init_weights()        

    def forward(self, x):
        b, c, h, w = x.size()

        ### encoding stage
        for e_block in self.e_blocks:
            x = e_block(x)

        ### reshaping
        z = z.view(b, -1)

        return self.m_head(z), self.v_head(v)

class Encoder16(BaseNetwork):
    """
    Bottom-up 16x16 encoder architecture
    """
    def __init__(
        self, 
        module_name='Encoder16',
        block_chns=[3, 128, 64, 16, 1],
        r=.01,
        use_spc_norm=False,
        init_weights=True
    ):
        super(Encoder16, self).__init__(module_name=module_name)

        ##### + encoding (down-sampling) layer
        self.ds_conv = nn.Sequential(
            DownBlock2x(
                in_channels=block_chns[0], 
                out_channels=block_chns[1], 
                r=r,
                use_spc_norm=use_spc_norm
            )            
        )
        
        self.backbone = Encoder8(
            module_name='Encoder8_in16',
            block_chns=block_chns[1:],
            r=r,
            use_spc_norm=use_spc_norm,
            init_weights=init_weights
        )

        if init_weights:
            self.init_weights()        

    def forward(self, x):
        b, c, h, w = x.size()

        ### encoding stage
        f = self.ds_conv(x)

        mu, logvar = self.backbone(f)

        return self.m_head(z), self.v_head(v)

class Encoder32(BaseNetwork):
    """
    Bottom-up 32x32 encoder architecture
    """
    def __init__(
        self, 
        module_name='Encoder32',
        block_chns=[3, 128, 64, 64, 16, 1],
        r=.01,
        use_spc_norm=False,
        init_weights=True
    ):
        super(Encoder32, self).__init__(module_name=module_name)

        ##### + encoding (down-sampling) layer
        self.ds_conv = nn.Sequential(
            DownBlock2x(
                in_channels=block_chns[0], 
                out_channels=block_chns[1], 
                r=r,
                use_spc_norm=use_spc_norm
            ),
            DownBlock2x(
                in_channels=block_chns[1], 
                out_channels=block_chns[2], 
                r=r,
                use_spc_norm=use_spc_norm
            )            
        )
        
        self.backbone = Encoder8(
            module_name='Encoder8_in32',
            block_chns=block_chns[2:],
            r=r,
            use_spc_norm=use_spc_norm,
            init_weights=init_weights
        )

        if init_weights:
            self.init_weights()        

    def forward(self, x):
        b, c, h, w = x.size()

        ### encoding stage
        f = self.ds_conv(x)

        mu, logvar = self.backbone(f)

        return self.m_head(z), self.v_head(v)

class UNet(BaseNetwork):
    """
    UNet architecture
    """
    def __init__(
        self, 
        output_dim,
        module_name='UNetDecoder',
        block_chns=[3 + 2 + 64, 128, 256, 128, 64],
        r=.01,
        use_var_head=False,
        use_spc_norm=False,
        init_weights=True
    ):
        super(UNet, self).__init__(module_name=module_name)

        self.use_var_head = use_var_head

        ##### + encoding blocks
        self.e_blocks = [
            SameBlockWOLReLU(
                block_chns[0], 
                block_chns[1], 
                r=r, 
                use_spc_norm=use_spc_norm
            )
        ]

        for i in range(1, len(block_chns) - 1, 1):
            self.e_blocks.append(
                SameBlockPreLReLU(
                    block_chns[i], 
                    block_chns[i + 1], 
                    r=r,
                    use_spc_norm=use_spc_norm
                )
            )

        self.e_blocks = nn.ModuleList(self.e_blocks)

        ##### + decoding blocks
        self.d_blocks = []

        for i in range(len(block_chns) - 1, 0, -1):            
            num_chn_cat = 0 if i == len(block_chns) - 1 \
                            else block_chns[i]
            out_chn = block_chns[i - 1] if i - 1 > 0 \
                                        else block_chns[-1]

            self.d_blocks.append(
                SameBlockPreLReLU(
                    block_chns[i] + num_chn_cat, 
                    out_chn,                    
                    r=r,
                    use_spc_norm=use_spc_norm
                )
            )

        self.d_blocks = nn.ModuleList(self.d_blocks)

        self.output_mean = Conv1x1Head(
            module_name='Conv1x1',
            in_channels=block_chns[-1], 
            out_channels=output_dim,
            use_bias=False,
            r=r,
            use_spc_norm=use_spc_norm,
            init_weights=init_weights
        )
        if use_var_head:
            self.output_var = Conv1x1Head(
                module_name='Conv1x1',
                in_channels=block_chns[-1], 
                out_channels=output_dim,
                use_bias=False,
                r=r,
                use_spc_norm=use_spc_norm,
                init_weights=init_weights
            )

        if init_weights:
            self.init_weights()        

        # # coordinate
        # x = torch.linspace(-1, 1, im_size)
        # y = torch.linspace(-1, 1, im_size)
        # x_grid, y_grid = torch.meshgrid(x, y)
        # # Add as constant, with extra dims for N and C
        # self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        # self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    def forward(self, x):
        # b, c, h, w = x.size()

        # # View z as 4D tensor to be tiled across new H and W dimensions
        # # Shape: NxDx1x1
        # z = z.view(z.shape + (1, 1))

        # # Tile across to match image size
        # # Shape: bxcxhxw
        # z = z.expand(-1, -1, h, w)

        # # Expand grids to batches and concatenate on the channel dimension
        # # Shape: bx(3+c+2)xhxw
        # x = torch.cat((self.x_grid.expand(b, -1, -1, -1),
        #                self.y_grid.expand(b, -1, -1, -1), z), dim=1)


        ### encoding stage
        f_stack = []
        for e_block in self.e_blocks:
            x = e_block(x)
            f_stack.append(x)

        ### decoding stage
        for i, (f, d_block) in enumerate(
            zip(f_stack[::-1], self.d_blocks)
        ):
            x = f if i == 0 \
                  else torch.cat([x, f], dim=1)
            x = d_block(x)

        if self.use_var_head:
            return self.output_mean(x), self.output_var(x)
        return self.output_mean(x)

###################################################################
########################## MISC. NETS #############################
###################################################################

class NonLinearHead(BaseNetwork):
    """
    One-layer 1x1 Conv. transformation
    """
    def __init__(
        self, 
        module_name='Linear',
        in_channels=64, 
        out_channels=64,       
        use_bias=False,
        r=.01,
        use_spc_norm=False,
        init_weights=True
    ):
        super(NonLinearHead, self).__init__(
            module_name=module_name
        )

        self.model = nn.Sequential(
            nn.LeakyReLU(
                r, 
                inplace=False # True
            ),
            spectral_norm(
                nn.Linear(
                    in_features=in_channels, 
                    out_features=out_channels,                    
                    bias=use_bias
                ),
                mode=use_spc_norm
            )
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        return self.model(x)

class Conv1x1Head(BaseNetwork):
    """
    One-layer 1x1 Conv. transformation
    """
    def __init__(
        self, 
        module_name='Conv1x1',
        in_channels=64, 
        out_channels=64,
        use_bias=False,
        r=.01,
        use_spc_norm=False,
        init_weights=True
    ):
        super(Conv1x1Head, self).__init__(
            module_name=module_name
        )

        self.model = nn.Sequential(
            nn.LeakyReLU(
                r, 
                inplace=False # True
            ),
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=1,
                    padding=0, 
                    bias=use_bias
                ),
                mode=use_spc_norm
            )
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        return self.model(x)

###################################################################
########################## LATENT EBMS ############################
###################################################################

class LEBMNet(BaseNetwork):
    def __init__(
        self, 
        module_name='LEBMNet',
        n_lat=64,
        n_emb=200,
        n_ds=10,
        r=.01,
        use_spc_norm=False,
        init_weights=True
    ):
        super(LEBMNet, self).__init__(module_name=module_name)         
        self.n_ds = n_ds

        self.model = nn.Sequential(
            spectral_norm(
                nn.Linear(
                    in_features=n_lat, 
                    out_features=n_emb
                ),
                mode=use_spc_norm
            ),
            nn.GELU(),
            spectral_norm(
                nn.Linear(
                    in_features=n_emb, 
                    out_features=n_emb
                ),
                mode=use_spc_norm
            ),
            nn.GELU(),
            spectral_norm(
                nn.Linear(
                    in_features=n_emb, 
                    out_features=n_emb
                ),
                mode=use_spc_norm
            ),
            nn.GELU(),            
            spectral_norm(
                nn.Linear(
                    in_features=n_emb, 
                    out_features=n_ds
                ),
                mode=use_spc_norm
            )
        )
                
        if init_weights:
            self.init_weights()

    def forward(self, z):
        b, h = z.size()
        logits = self.model(z).view(b, -1) # (b, n_lat) -> (b, n_ds)

        score = torch.logsumexp(logits, dim=1, keepdim=True)
        return score.view(b, 1), logits.view(b, self.n_ds)
