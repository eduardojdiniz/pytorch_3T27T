from abc import ABC, abstractmethod, ABCMeta
import functools
from skimage import color  # used for lab2rgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_3T27T.base import BaseNet
from pytorch_3T27T.utils import setup_logger

logger = setup_logger(__name__)


__all__ = ['Pix2Pix', 'ColorizationPix2Pix' 'CycleGAN', 'get_discriminator',
           'get_generator']



class Generator(BaseNet, metaclass=ABCMeta):
    """Generic Generator"""

    def __init__(self, in_ch, out_ch, n_filters, n_blocks, norm_layer,
                 use_dropout, verbose):
        """
        Define a Generator

        Parameters
        ----------
        in_ch : int
            The number of channels in input images
        out_ch : int
            The number of channels in output images
        n_filters : int
            The number of filters in the last convolutional layer
        n_blocks : int
            The number of intermediate blocks
        norm_layer : torch.nn or functools.partial
            Normalization layer
        use_dropout : bool
            If True, use dropout layers
        verbose : bool
            If true, prints the net's torchinfo summary
        """

        assert(n_blocks >= 0)

        super(Generator, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.verbose = verbose

        # If norm_layer == nn.BatchNorm2d, no need to use bias since
        # batch normalization has affine parameters,
        # norm_layer == nn.Instancenorm2d, then use bias
        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d

        self.net = self.build_net()


    @abstractmethod
    def build_net(self): -> nn.Sequential
        """Build the Network"""


    @abstractmethod
    def forward(self, x):
        """Forward pass through the Network"""



class ResNetGenerator(Generator):
    """
    Resnet-based generator that consists of Resnet blocks between a few
    downsampling/upsampling operations.
    """

    def __init__(self, in_ch, out_ch, n_filters, n_blocks, padding_type,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, verbose=False):
        """
        Construct a Resnet-based generator

        Parameters
        ----------
        in_ch : int
            The number of channels in input images
        out_ch : int
            The number of channels in output images
        n_filters : int
            The number of filters in the last convolutional layer
        n_blocks : int
            The number of ResNet blocks
        norm_layer : torch.nn or functools.partial
            Normalization layer
        use_dropout : bool
            If True, use dropout layers
        padding_type : str
            The name of padding layer in convolutional layers.
            One of reflect | replicate | zero
        verbose : bool
            If true, prints the net's torchinfo summary
        """

        super(ResNetGenerator, self).__init__(in_ch, out_ch, n_filters,
                                              n_blocks, norm_layer,
                                              use_dropout, verbose)

        self.padding_type = padding_type

        logger.info('<init>: \n %s', self)


    def build_net(self): -> nn.Sequential
        """Creates a ResNet-based Generator"""

        net = []

        # Add first Block
        net.extend(self.first_block())

        # Add downsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            net.extend(self.downsampling(mult * self.n_filters))

        # Add ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            net.append(ResNetBlock(mult * self.n_filters,
                                     padding_type=self.padding_type,
                                     norm_layer=self.norm_layer,
                                     use_dropout=self.use_dropout,
                                     use_bias=self.use_bias)]

        # Add upsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            net.extend(self.upsampling(mult * self.n_filters))

        # Add last block
        net.extend(self.last_block())

        return nn.Sequential(*net)


    def first_block(self):
        block = [nn.ReflectionPad2d(3)]
        block += [nn.Conv2d(self.in_ch, self.n_filters, kernel_size=7,
                            padding=0, bias=self.use_bias)]
        block += [self.norm_layer(self.n_filters)]
        block += [nn.ReLU(True)]
        return block


    def downsampling(self, n_filters):
        block = [nn.Conv2d(n_filters, 2 * n_filters, kernel_size=3, stride=2,
                           padding=1, bias=self.use_bias)]
        block += [self.norm_layer(2 * n_filters)]
        block += [nn.ReLU(True)]
        return block


    def upsampling(self, n_filters):
        block = [nn.ConvTranspose2d(n_filters, int(n_filters / 2),
                                     kernel_size=3, stride=2, padding=1,
                                     output_padding=1, bias=self.use_bias)]
        block += [self.norm_layer(int(n_filters / 2))]
        block += [nn.ReLU(True)]
        return block


    def last_block(self):
        block = [nn.ReflectionPad2d(3)]
        block += [nn.Conv2d(self.n_filters, self.out_ch, kernel_size=7,
                            padding=0)]
        block += [nn.Tanh()]
        return block



class ResNetBlock(Generator):
    """Define a ResNet Block"""

    def __init__(self, dim, padding_type, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        """
        Initialize the ResNet block
        A ResNet block is a convolutional block with skip connections
        The original ResNet paper: https://arxiv.org/pdf/1512.03385v1.pdf

        Parameters
        ----------
        dim : int
            The number of channels in the convolutional layer
        padding_type : str
            The name of padding layer in convolutional layers.
            One of reflect | replicate | zero
        norm_layer : torch.nn or functools.partial
            Normalization layer
        use_dropout : bool
            If True, use dropout layers
        use_bias : bool
            If True, add bias to the convolutional layer
        """

        super(ResNetBlock, self).__init__(dim, dim, None, None, norm_layer,
                                          use_dropout, False)

        self.use_padding_layer = True
        self.conv_padding = 0
        if padding_type == 'reflect':
            self.padding_layer = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.padding_layer = nn.ReplicationPad2d(1)
        elif padding_type == 'zero':
            self.use_padding_layer = False
            self.conv_padding = 1
        else
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        logger.info('<init>: \n %s', self)


    def build_net(self):
        block = []

        if self.use_padding_layer:
            block += [self.padding_layer]

        block += [nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3,
                            padding=self.conv_padding, bias=self.use_bias)]
        block += [self.norm_layer(self.dim)]
        block += [nn.ReLU(True)]

        if self.use_dropout:
            block += [nn.Dropout(0.5)]

        if self.use_padding_layer:
            block += [self.padding_layer]

        block += [nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3,
                            padding=self.conv_padding, bias=self.use_bias)]
        block += [self.norm_layer(self.dim)]

        return nn.Sequential(*block)


    def forward(self, x):
        """Forward function add skip connection to the convolutional block"""

        return self.net(x) + x



class UNetGenerator(Generator):
    """Define a UNet-based Generator"""

    def __init__(self, in_ch, out_ch, n_filters, n_blocks,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, verbose=False):
        """
        Initialize the UNet-based Generator

        Parameters
        ----------
        in_ch : int
            The number of channels in input images
        out_ch : int
            The number of channels in output images
        n_filters : int
            The number of filters in the last convolutional layer
        n_blocks : int
            The number of downsampling layers in UNet
            For example, if |n_blocks| == 7, image of size 128x128 will
            become of size 1x1 at the bottleneck
        norm_layer : torch.nn or functools.partial
            Normalization layer
        use_dropout : bool
            If True, use dropout layers
        verbose : bool
            If true, prints the net's torchinfo summary
        """

        super(UNetGenerator, self).__init__(in_ch, out_ch, n_filters,
                                            n_blocks, norm_layer,
                                            use_dropout, verbose)

        logger.info('<init>: \n %s', self)


    def build_net(self):
        """
        Build a UNet-based Generator
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.

        Returns
        -------
            A UNetSkipConnectionBlock object
        """

        net = []

        # Add the innermost layer
        unet_block = UNetSkipConnectionBlock(8 * self.n_filters,
                                             8 * self.n_filters,
                                             in_ch=None,
                                             submodule=None,
                                             norm_layer=self.norm_layer,
                                             innermost=True)

        # Add intermediate layers with (8 * n_filters) filters
        for i in range(self.n_blocks - 5):
            unet_block = UNetSkipConnectionBlock(8 * self.n_filters,
                                                 8 * self.n_filters,
                                                 in_ch=None,
                                                 submodule=unet_block,
                                                 norm_layer=self.norm_layer,
                                                 use_dropout=self.use_dropout)

        # Gradually reduce the num of filters from (8 * n_filters) to n_filters
        unet_block = UNetSkipConnectionBlock(4 * self.n_filters,
                                             8 * self.n_filters,
                                             in_ch=None,
                                             submodule=unet_block,
                                             norm_layer=self.norm_layer)

        unet_block = UNetSkipConnectionBlock(2 * self.n_filters,
                                             4 * self.n_filters,
                                             in_ch=None,
                                             submodule=unet_block,
                                             norm_layer=self.norm_layer)

        unet_block = UNetSkipConnectionBlock(1 * self.n_filters,
                                             2 * self.n_filters,
                                             in_ch=None,
                                             submodule=unet_block,
                                             norm_layer=self.norm_layer)

        # Add the outermost layer
        net = UNetSkipConnectionBlock(self.out_ch,
                                      self.n_filters,
                                      in_ch=self.in_ch,
                                      submodule=unet_block,
                                      norm_layer=self.norm_layer,
                                      outermost=True)
        return net


    def forward(self, x):
        """Forward pass"""
        return self.net(x)



class UNetSkipConnectionBlock(Generator):
    """
    U-Net block with skip connection
    X────────────────────identity─────────────────>
    └──>|downsampling|─>|submodule|─>|upsampling|─>
    """

    def __init__(self, outer_ch, inner_ch, in_ch=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        """
        Define a U-Net block with skip connection

        Parameters
        ----------
        outer_ch : int
            The number of filters in the outer convolutional layer
        inner_ch : int
            The number of filters in the inner convolutional layer
        in_ch : int
            The number of channels in input images
        submodule : UNetSkipConnectionBlock
            A previously defined U-Net block with skip connection
        outermost : bool
            If True, this is the outermost layer
        innermost : bool
            If True, this is the innermost layer
        norm_layer : torch.nn or functools.partial
            Normalization layer
        use_dropout : bool
            If True, use dropout layers
        """

        if in_ch is None:
            in_ch = outer_ch
        super(UNetSkipConnectionBlock, self).__init__(in_ch, outer_ch,
                                                      inner_ch, None,
                                                      norm_layer, use_dropout,
                                                      False)
        self.submodule = submodule
        self.outermost = outermost
        self.innermost = innermost

        logger.info('<init>: \n %s', self)


    def build_net(self): -> nn.Sequential
        if self.outermost:
            return self.build_outermost()
        elif self.innermost:
            return self.build_innemost()
        else:
            return self.build_intermediate()


    def build_outermost(self):
        up = nn.ConvTranspose2d(2 * self.n_filters, self.out_ch,
                                kernel_size=4, stride=2, padding=1)
        down = nn.Conv2d(self.in_ch, self.n_filters, kernel_size=4,
                         stride=2, padding=1, bias=self.use_bias)
        block = [down]
        block += [self.submodule]
        block += [nn.ReLU(True), up, nn.Tanh()]

        return nn.Sequential(*block)


    def build_innermost(self):
        up = nn.ConvTranspose2d(self.n_filters, self.ou_ch, kernel_size=4,
                                stride=2, padding=1, bias=self.bias)
        down = nn.Conv2d(self.in_ch, self.n_filters, kernel_size=4,
                         stride=2, padding=1, bias=self.use_bias)

        block = [nn.LeakyReLU(0.2, True), down]
        block += [nn.ReLU(True), up, self.norm_layer(self.out_ch)]

        return nn.Sequential(*block)


    def build_intermediate(self):
        up = nn.ConvTranspose2d(2 * self.n_filters, self.out_ch,
                                kernel_size=4, stride=2, padding=1,
                                bias=self.bias)
        down = nn.Conv2d(self.in_ch, self.n_filters, kernel_size=4,
                         stride=2, padding=1, bias=self.use_bias)

        block = [nn.LeakyReLU(0.2, True), down,
                 self.norm_layer(self.n_filters)]
        block += [self.submodule]
        block += [nn.ReLU(True), up, self.norm_layer(self.out_ch)]

        if self.use_dropout:
            block += [nn.Dropout(0.5)]

        return nn.Sequential(*block)


    def forward(self, x):
        """Forward pass"""

        if self.outermost:
            return self.net(x)
        # Add skip connection
        else:
            return torch.cat([x, self.net(x)], 1)



class Discriminator(BaseNet, metaclass=ABCMeta):
    """Generic Discriminator"""

    def __init__(self, in_ch, out_ch, kernel_size, padding_size, n_layers,
                 norm_layer, verbose):
        """
        Define a Discriminator

        Parameters
        ----------
        in_ch : int
            The number of channels in input images
        out_ch : int
            The number of channels in output images
        kernel_size : int
            Size of the convolving kernel
        padding_size : int
            Padding added to all four sides of the input
        n_layers : int
            The number of convolutional layers in the discriminator
        norm_layer : torch.nn or functools.partial
            Normalization layer
        verbose : bool
            If true, prints the net's torchinfo summary
        """

        assert(n_layers >= 0)

        super(Discriminator, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.n_layers = n_layers
        self.norm_layer = norm_layer
        self.verbose = verbose

        # If norm_layer == nn.BatchNorm2d, no need to use bias since
        # batch normalization has affine parameters,
        # else norm_layer == nn.Instancenorm2d, then use bias
        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d

        self.net = self.build_net()


    @abstractmethod
    def build_net(self): -> nn.Sequential
        """Build the Network"""


    @abstractmethod
    def forward(self, x):
        """Forward pass through the Network"""



class PatchDiscriminator(Discriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_ch, out_ch=64, kernel_size=4, padding_size=1,
                 n_layers=3, norm_layer=nn.BatchNorm2d, verbose=False):
        """
        Construct a PatchGAN discriminator
        Classifies if a patch of the image is real or not.

        Parameters
        ----------
        in_ch : int
            The number of channels in input images
        out_ch : int
            The number of channels in the last convolutional layer
        n_layers : int
            The number of convolutional layers in the discriminator
        norm_layer : torch.nn or functools.partial
            Normalization layer
        verbose : bool
            If true, prints the net's torchinfo summary
        """

        super(PatchDiscriminator, self).__init__(in_ch, out_ch, kernel_size,
                                                 padding_size, n_layers,
                                                 norm_layer, verbose)

        logger.info('<init>: \n %s', self)


    def build_net(self): -> nn.Sequential
        """
        Create the PatchGAN Network

        Returns
        -------
            A nn.Sequential pytorch sequential container
        """

        net = []

        # Add first Block
        net.extend(self.first_block())

        # Add downsampling layers
        prev_mult = 1
        curr_mult = 1
        for n in range(1, self.n_layers):
            prev_mult = curr_mult
            curr_mult = min(2 ** n, 8)
            _in_ch = prev_mult * self.out_ch
            _out_ch = curr_mult * self.out_ch
            net.extend(self.downsampling(_in_ch, _out_ch, stride=2))

        prev_mult = curr_mult
        curr_mult = min(2 ** self.n_layers, 8)
        _in_ch = prev_mult * self.out_ch
        _out_ch = curr_mult * self.out_ch
        net.extend(self.downsampling(_in_ch, _out_ch, stride=1))

        # Add last block
        net.extend(self.last_block())

        return nn.Sequential(*net)


    def first_block(self):
        block = [nn.Conv2d(self.in_ch, self.out_ch,
                           kernel_size=self.kernel_size, stride=2,
                           padding=self.padding_size)]
        block += [nn.LeakyReLU(0.2, True)]
        return block


    def downsampling(self, in_ch, out_ch, stride):
        block = [nn.Conv2d(in_ch, out_ch, kernel_size=self.kernel_size,
                           stride=stride, padding=self.padding_size,
                           bias=self.use_bias)]
        block += [self.norm_layer(out_ch)]
        block += [nn.LeakyReLU(0.2, True)]
        return block


    def last_block(self, in_ch):
        block = [nn.Conv2d(self.in_ch, 1, kernel_size=self.kernel_size,
                           stride=1, padding=self.padding_size)]
        return block


    def forward(self, x):
        """Forward pass"""

        return self.net(x)



class PixelDiscriminator(Discriminator):
    """Defines a 1x1 PatchGAN discriminator (PixelGAN)"""

    def __init__(self, in_ch, out_ch=64, norm_layer=nn.BatchNorm2d,
                 verbose=False):
        """
        Construct a PixelGAN discriminator
        A PixelGAN is a 1x1 PatchGAN with padding equal to 0,
        stride equal to 1, and number of layers equal to 1.
        Classifies if each pixel is real or not.

        Parameters
        ----------
        in_ch : int
            The number of channels in input images
        out_ch : int
            The number of channels in the last convolutional layer
        norm_layer : torch.nn or functools.partial
            Normalization layer
        verbose : bool
            If true, prints the net's torchinfo summary
        """

        super(PixelDiscriminator, self).__init__(in_ch, out_ch, 1, 0, 1,
                                                 norm_layer, verbose)

        logger.info('<init>: \n %s', self)


    def build_net(self): -> nn.Sequential
        """Create the PixelGAN Network"""

        net = []

        # Add first Block
        net += [nn.Conv2d(self.in_ch, self.out_ch,
                            kernel_size=self.kernel_size, stride=1,
                            padding=self.padding_size)]
        net += [nn.LeakyReLU(0.2, True)]

        # Add middle Block
        net = [nn.Conv2d(self.out_ch, 2 * self.out_ch,
                           kernel_size=self.kernel_size, stride=1,
                           padding=self.padding_size, bias=self.use_bias)]
        net += [self.norm_layer(2 * self.out_ch)]
        net += [nn.LeakyReLU(0.2, True)]

        # Add last block
        net = [nn.Conv2d(2 * self.out_ch, 1, kernel_size=self.kernel_size,
                           stride=1, padding=self.padding_size)]

        return nn.Sequential(*net)


    def forward(self, x):
        """Forward pass"""

        return self.net(x)



class GAN(BaseNet, metaclass=ABCMeta):
    """Generic GAN"""

    def __init__(self, generator, discriminator, verbose, **gen_kwargs,
                 **disc_kwargs):

        """
        Define a GAN

        Parameters
        ----------
        generator : str
            The generator name
            One of: unet_128 | unet_256 | resnet_6blocks | resnet_9blocks
        discriminator : str
            The discriminator name
            One of: patch | nlayer_patch | pixel
        verbose : bool
            If true, prints the net's torchinfo summary
        gen_kwargs : dict
            Generator parameters
        disc_kwargs : dict
            Discriminator parameters
        """

        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.verbose = verbose
        self.gen_kwargs = gen_kwargs
        self.disc_kwargs = disc_kwargs

        self.build_net()


    @abstractmethod
    def build_net(self):
        """Specify both Generators and Discriminators"""


    @abstractmethod
    def forward(self):
        """Forward pass through Generator(s)"""



class CycleGAN(GAN):
    """
    Defines a CycleGAN architecture, for learning image-to-image translation
    without paired data. By default, it uses a ResNet generator with 9 blocks
    and a PatchGAN discriminator, introduced by pix2pix.
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, generator='resnet_9blocks', discriminator='patch',
                 verbose=False, **gen_kwargs, **disc_kwargs):

        """
        Initialize the CycleGAN architecture

        Parameters
        ----------
        generator : str
            The generator's name
            One of: unet_128 | unet_256 | resnet_6blocks | resnet_9blocks
        discriminator : str
            The discriminator's name
            One of: patch | nlayer_patch | pixel
        verbose : bool
            If true, prints the net's torchinfo summary
        gen_kwargs : dict
            Generator parameters
        disc_kwargs : dict
            Discriminator parameters
        """
        super(GAN, self).__init__(generator, discriminator, verbose,
                                  **gen_kwargs, **disc_kwargs)

        logger.info('<init>: \n %s', self)


    def build_net(self):
        """
        Naming convention:
            Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        Specify both Generators and Discriminators
        """

        self.G_A = get_generator(self.generator **self.gen_kwargs)
        self.G_B = get_generator(self.generator **self.gen_kwargs)
        self.D_A = get_discriminator(self.discriminator, **self.disc_kwargs)
        self.D_B = get_discriminator(self.discriminator **self.disc_kwargs)


    def forward(self, real_A, real_B):
        """Forward pass through generator(s)"""

        fake_B = self.G_A(real_A)  # G_A(A)
        recon_A = self.G_B(fake_B) # G_B( G_A(A) )
        fake_A = self.G_B(real_B)  # G_B(B)
        recon_B = self.G_A(fake_A) # G_A( G_B(B) )
        return fake_B, recon_A, fake_A, recon_B



class Pix2Pix(GAN):
    """
    Defines a Pix2Pix architecture, for learning a mapping from input images to
    output images given paired data. By default, it uses a U-Net generator and
    a PatchGAN discriminator.
    Pix2Pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, generator='unet_256', discriminator='patch',
                 verbose=False, **gen_kwargs, **disc_kwargs):

        """
        Initialize the Pix2Pix architecture

        Parameters
        ----------
        generator : str
            The generator's name
            One of: unet_128 | unet_256 | resnet_6blocks | resnet_9blocks
        discriminator : str
            The discriminator's name
            One of: patch | nlayer_patch | pixel
        verbose : bool
            If true, prints the net's torchinfo summary
        gen_kwargs : dict
            Generator Parameters
        disc_kwargs : dict
            Discriminator Parameters
        """

        # Conditional GANs need to take both input and output images;
        # Therefore, the number of channels for the discriminator is the number
        # of channels in the input plus the number of channels in the output
        disc_kwargs['out_ch'] = gen_kwargs['in_ch'] + gen_kwargs['out_ch']

        super(GAN, self).__init__(generator, discriminator, verbose,
                                  **gen_kwargs, **disc_kwargs)

        logger.info('<init>: \n %s', self)


    def build_net(self):
        """Specify both Generator and Discriminator"""

        self.G = get_generator(self.generator **self.gen_kwargs)
        self.D = get_discriminator(self.discriminator, **self.disc_kwargs)


    def forward(self, real):
        """Forward pass through generator"""

        fake = self.G(real)  # G(A)
        return fake


class ColorizationPix2Pix(GAN):
    """
    Defines a Pix2Pix architecture for image colorization, i.e.,
    black & white images -> colorful images. By default, it uses a U-Net
    generator and a PatchGAN discriminator. It trains a Pix2Pix net, mapping
    from L channel to ab channels in Lab color space. By default, the number of
    channels in the input image is 1 and the number of channels in the output
    image is 2.
    """

    def __init__(self, generator='unet_256', discriminator='patch',
                 verbose=False, **gen_kwargs, **disc_kwargs):

        """
        Initialize the Colorization Pix2Pix architecture

        Parameters
        ----------
        generator : str
            The generator's name
            One of: unet_128 | unet_256 | resnet_6blocks | resnet_9blocks
        discriminator : str
            The discriminator's name
            One of: patch | nlayer_patch | pixel
        verbose : bool
            If true, prints the net's torchinfo summary
        gen_kwargs : dict
            Generator Parameters
        disc_kwargs : dict
            Discriminator Parameters
        """

        # Conditional GANs need to take both input and output images;
        # Therefore, the number of channels for the discriminator is the number
        # of channels in the input plus the number of channels in the output
        gen_kwargs['in_ch'] = 1 # Black & white image
        gen_kwargs['out_ch'] = 2 # Lab color space
        disc_kwargs['out_ch'] = gen_kwargs['in_ch'] + gen_kwargs['out_ch']

        super(GAN, self).__init__(generator, discriminator, verbose,
                                  **gen_kwargs, **disc_kwargs)

        logger.info('<init>: \n %s', self)


    def build_net(self):
        """Specify both Generator and Discriminator"""

        self.G = get_generator(self.generator **self.gen_kwargs)
        self.D = get_discriminator(self.discriminator, **self.disc_kwargs)


    def forward(self, real):
        """Forward pass through generator"""

        fake = self.G(real)  # G(A)
        return fake


    def lab2rgb(self, L: torch.Tensor, AB: torch.Tensor): -> np.array
        """
        Convert an Lab tensor image to a RGB numpy output

        Parameters
        ----------
        L :  torch.Tensor
            L channel images. 1-channel tensor array, range: [-1, 1]
        AB : torch.Tensor
            ab channel images. 2-channel tensor array, range: [-1, 1]

        Returns
        -------
        rgb : np.array
            RGB output images. NumPy image images, range: [0, 255]
        """

        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb


def get_generator(generator: str, **gen_kwargs): -> Generator
    """
    Construct a Generator based on the supplied name

    Parameters
    ----------
    generator : str
        The generator's name
        One of: unet_128 | unet_256 | resnet_6blocks | resnet_9blocks
    gen_kwargs : dict
        Parameters to the Generator constructor
    """

    n_blocks = kwargs.pop('n_blocks', None)

    generators = {
        "unet_128": UNetGenerator(n_blocks=7, **gen_kwargs)
        "unet_256": UNetGenerator(n_blocks=8, **gen_kwargs)
        "resnet_6blocks": ResNetGenerator(n_blocks=6, **gen_kwargs)
        "resnet_9blocks": ResNetGenerator(n_blocks=9, **gen_kwargs)
    }

    if generator in generators:
        return generators[generator]
    else:
        msg = (f"Generator Network name [{generator}] "
               f"is not recognized")
        raise NotImplementedError(msg)


def get_discriminator(discriminator: str, **disc_kwargs): -> Discriminator
    """
    Construct a Discriminator based on the supplied name

    Parameters
    ----------
    discriminator : str
        The discriminator's name
        One of: patch | nlayer_patch | pixel
    kwargs : dict
        Parameters to the Discriminator constructor
    """

    in_ch = kwargs.pop('in_ch', 3) # Default to colored image
    n_layers = kwargs.pop('n_layers', 3) # Default to 3 layers

    discriminators = {
        "patch": PatchDiscriminator(in_ch, n_layers=3, **disc_kwargs),
        "nlayer_patch": PatchDiscriminator(in_ch, n_layers=n_layers,
                                           **disc_kwargs),
        "pixel": PixelDiscriminator(in_ch, **disc_kwargs)
    }

    if discriminator in discriminators:
        return discriminators[discriminator]
    else:
        msg = (f"Discriminator Network name [{discriminator}] "
               f"is not recognized")
        raise NotImplementedError(msg)
