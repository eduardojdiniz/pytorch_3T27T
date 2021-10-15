#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_3T27T.base import BaseLoss


logger = setup_logger(__name__)


__all__ = ['WGANGLoss', 'Pix2PixLoss', 'CycleGANLoss']



class WGANGPLoss(BaseLoss):
    """
    Implement the gradient penalty loss.
    WGAN-GP paper https://arxiv.org/abs/1704.00028
    """

    def __init__(self, data_type: str = "mixed", constant=1.0, lambda_gp=10.0,
                 weight=None, reduction='mean'):
        """
        Arguments
        ---------
        data_type : str
            Specify if we use real images, fake images or a linear
            interpolation of the two.

        constant : float
            The constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp : float
            Weight for this loss
        """

        super(GANLoss, self).__init__(weight=weight, reduction=reduction)

        self.data_type = data_type
        self.constant = constant
        self.lambda_gp = lambda_gp

        logger.info('<init>: \n %s', self)


    def forward(self, netD, real, fake): -> torch.Tensor
        """
        Computes the gradient penalty loss

        Parameters
        ----------
        netD : torch.device
            Discriminator network
        real : torch.Tensor
            Real images
        fake : torch.Tensor
            Fake images
        target : torch.Tensor
             Discriminator prediction for output

        Returns
        -------
        Gradient penalty loss
        """

        output = self.get_output_from_type(real, fake)
        target = netD(output)

        if self.lambda_gp > 0.0:
            output.requires_grad_(True)
            grad_outputs = torch.ones(target.size()).to(output.device)
            gradients = torch.autograd.grad(outputs=target, inputs=output,
                                            grad_outputs=grad_outputs,
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)

            # Flatten the data
            gradients = gradients[0].view(output.size(0), -1)
            loss = ( (gradients + 1e-16).norm(2, dim=1) - constant) ** 2
            loss = loss.mean() * lambda_gp
            return loss
        else:
            return 0.0


    def get_output_from_type(self, real, fake): -> torch.Tensor
        """
        Get output parameter for forward passed based on data_type

        Parameters
        ----------
        real : torch.Tensor
            Real images
        fake : torch.Tensor
            Fake images

        Returns
        -------
        output : torch.Tensor
            Interpolated data given data_type
        """
        batch_size = real.shape[0]
        if self.data_type == 'real':
            output = real
        elif self.data_type == 'fake':
            output = fake
        elif self.type == 'mixed':
            alpha = torch.rand(batch_size, 1).to(real.device)
            alpha = alpha.expand(batch_size, real.nelement() //
                                 batch_size).contiguous().view(*real.shape)
            output = alpha * real + ((1 - alpha) * fake)
        else:
            raise NotImplementedError(f'{self.data_type} not implemented')

        return output



class Pix2PixLoss(GANLoss):
    """Define the Pix2Pix GAN Loss for both the discriminator and generator"""

    def __init__(self, lambda_L1, GAN_criterion='vanilla',
                 target_real_label=1.0, target_fake_label=0.0,
                 weight=None, reduction='mean'):
        """
        Parameters
        ----------
        lambda_L1 : torch.Tensor
            The L1 regularization weight
        GAN_criterion : str
            The type of GAN criterion. Default: vanilla
        target_real_label : bool
            label for a real image. Default: 1.0
        target_fake_label : bool
            label for a fake image. Default: 0.0
        weight : torch.Tensor (optional)
            A manual rescaling weight given to the loss of each batch element.
            If given, has to be a Tensor of size nbatch. Default: None
        reduction : str (optional)
            Specifies the reduction to apply to the output: 'none' | 'mean' |
            'sum'. 'none': no reduction will be applied, 'mean': the sum of the
            output will be divided by the number of elements in the output,
            'sum': the output will be summed. Note: size_average and reduce are
            in the process of being deprecated, and in the meantime, specifying
            either of those two args will override reduction. Default: 'mean'
        """

        super(Pix2PixLoss, self).__init__(GAN_criterion,
                                          target_real_label=target_real_label,
                                          target_fake_label=target_fake_label,
                                          weight=weight, reduction=reduction)

        self.L1Criterion = nn.L1Loss()
        self.lambda_L1 = lambda_L1


    def forward_G(self):
        """
        Computes GAN and L1 loss for the generator

        Parameters
        ----------
        netD : torch.device
            Discriminator network
        real_A : torch.Tensor
            Real images from domain A
        real_B : torch.Tensor
            Real images from domain B
        fake_B : torch.Tensor
            Fake images from domain B

        Returns
        -------
        loss_G : torch.Tensor
            total generator loss
        """

        ## First, Generator should fake the discriminator

        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB)
        target_fake = self.get_target_tensor(pred_fake, True)
        loss_G_GAN = self.GANCriterion(pred_fake, target_fake)

        ## Second G(A) = B
        loss_G_L1 = self.L1Criterion(fake_B, real_B)

        # Combine losses
        loss_G = loss_G_GAN + self.lambda_L1 * loss_G_L1
        return loss_G


    def forward_D(self, netD, real_A, real_B, fake_B):
        """
        Computes the discriminator loss

        Parameters
        ----------
        netD : torch.device
            Discriminator network
        real_A : torch.Tensor
            Real images from domain A
        real_B : torch.Tensor
            Real images from domain B
        fake_B : torch.Tensor
            Fake images from domain B

        Returns
        -------
        loss_D : torch.Tensor
            total discriminator loss
        """

        ## Fake

        # We use conditional GANs; we need to feed both input and output to the
        # discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)

        # Stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake_AB.detach())
        target_fake = self.get_target_tensor(pred_fake, False)
        loss_D_fake = self.GANCriterion(pred_fake, target_fake)

        ## Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB)
        target_real = self.get_target_tensor(pred_real, True)
        loss_D_real = self.GANCriterion(pred_real, target_real)

        # Combine losses
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D


    def forward(self, netD, real_A, real_B, fake_B, generator):
        """Compute either discriminator loss or generator loss"""
        if generator:
            return self.forward_G(netD, real_A, real_B, fake_B)
        else:
            return self.forward_D(netD, real_A, real_B, fake_B)



class CycleGANLoss(GANLoss):
    """Define the Cycle GAN Loss for all the discriminators and generators"""

    def __init__(self, lambda_idt, lambda_A, lambda_B, GAN_criterion='vanilla',
                 target_real_label=1.0, target_fake_label=0.0,
                 weight=None, reduction='mean'):
        """
        Parameters
        ----------
        lambda_idt : torch.Tensor
            The Identity loss weight
        lambda_A : torch.Tensor
            The Cycle real_A -> fake_B -> rec_A loss weight
        lambda_B : torch.Tensor
            The Cycle real_B -> fake_A -> rec_B loss weight
        GAN_criterion : str
            The type of GAN criterion. Default: vanilla
        target_real_label : bool
            label for a real image. Default: 1.0
        target_fake_label : bool
            label for a fake image. Default: 0.0
        weight : torch.Tensor (optional)
            A manual rescaling weight given to the loss of each batch element.
            If given, has to be a Tensor of size nbatch. Default: None
        reduction : str (optional)
            Specifies the reduction to apply to the output: 'none' | 'mean' |
            'sum'. 'none': no reduction will be applied, 'mean': the sum of the
            output will be divided by the number of elements in the output,
            'sum': the output will be summed. Note: size_average and reduce are
            in the process of being deprecated, and in the meantime, specifying
            either of those two args will override reduction. Default: 'mean'
        """

        super(CycleGANLoss, self).__init__(GAN_criterion,
                                           target_real_label=target_real_label,
                                           target_fake_label=target_fake_label,
                                           weight=weight, reduction=reduction)

        self.CycleCriterion = nn.L1Loss()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.IdtCriterion = nn.L1Loss()
        self.lambda_Idt = lambda_Idt


    def forward_G(self, netD_A, netD_B, real_A, real_B, fake_A, fake_B,
                  rec_A, rec_B, idt_A, idt_B):
        """
        Computes GAN, Cycle and Identity losses for the generators

        Parameters
        ----------
        netD_A : torch.device
            Domain A discriminator network, netD_A: G_B(B) vs. A
        netD_B : torch.device
            Domain B discriminator network, netD_B: G_A(A) vs. B
        real_A : torch.Tensor
            Real images from domain A
        real_B : torch.Tensor
            Real images from domain A
        fake_A : torch.Tensor
            Fake images from domain A, G_B(B)
        fake_B : torch.Tensor
            Fake images from domain B, G_A(A)
        rec_A : torch.Tensor
            Reconstructed images from domain A, G_B(G_A(A))
        rec_B : torch.Tensor
            Reconstructed images from domain A, G_A(G_B(B))
        idt_A : torch.Tensor
            Identity images from domain B, G_A(B)
        idt_B : torch.Tensor
            Identity images from domain A, G_B(A)

        Returns
        -------
        loss_G : torch.Tensor
            total generation loss
        """

        ## First, Generators should fake the discriminators
        # GAN loss D_A(G_A(A))
        pred_fake_B = netD_A(fake_B)
        target_true = self.get_target_tensor(pred_fake_B, True)
        loss_GAN_G_A = self.GANCriterion(pred_fake_B, target_true)
        # GAN loss D_B(G_B(B))
        pred_fake_A = netD_B(fake_A)
        target_true = self.get_target_tensor(pred_fake_A, True)
        loss_GAN_G_B = self.GANCriterion(pred_fake_A, target_true)

        ## Second, we ensure cycle consistency
        # Forward cycle loss || G_B(G_A(A)) - A ||
        loss_cycle_A = self.CycleCriterion(rec_A, real_A)
        # Backward cycle loss || G_A(G_B(B)) - B ||
        loss_cycle_B = self.CycleCriterion(rec_B, real_B)

        # Third, we ensure identity mapping
        if self.lambda_idt > 0
            # G_A should be identity if real_B is fed: "||G_A(B) - B||"
            loss_idt_A = self.IdtCriterion(self.idt_A, real_B)
            # G_B should be identity if real_A is fed: "||G_B(A) - A||"
            loss_idt_B = self.IdtCriterion(self.idt_B, real_A)
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # Combine losses
        loss_G = 0
        loss_G += loss_idt_A * self.lambda_B * self.lambda_Idt
        loss_G += loss_idt_B * self.lambda_A * self.lambda_Idt
        loss_G += loss_GAN_G_A
        loss_G += loss_GAN_G_B
        loss_G += loss_cycle_A * self.lambda_A
        loss_G += loss_cycle_B * self.lambda_B
        return loss_G


    def forward_D(self, netD, real, fake):
        """
        Computes the discriminators losses

        Parameters
        ----------
        netD : torch.device
            Discriminator network, netD: G(real) vs. real
        real : torch.Tensor
            Real images
        fake : torch.Tensor
            Fake images, G(real)

        Returns
        -------
        loss_D : torch.Tensor
            total discriminator loss
        """

        ## Real
        pred_real = netD(real)
        target_real = self.get_target_tensor(pred_real, True)
        loss_D_real = self.GANCriterion(pred_real, target_real)

        ## Fake
        # Stop backprop to the generator by detaching fake
        pred_fake = netD(fake.detach())
        target_fake = self.get_target_tensor(pred_fake, False)
        loss_D_fake = self.GANCriterion(pred_fake, target_fake)

        # Combine losses
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D


    def forward_D_A(self, netD_A, real_B, fake_B, fake_B_pool):
        """
        Computes GAN loss for discriminator A

        Parameters
        ----------
        netD_A : torch.device
            Domain A discriminator network, netD_A: G_B(B) vs. A
        netD_B : torch.device
            Domain B discriminator network, netD_B: G_A(A) vs. B
        real_A : torch.Tensor
            Real images from domain A
        real_B : torch.Tensor
            Real images from domain A
        fake_A : torch.Tensor
            Fake images from domain A, G_B(B)
        fake_B : torch.Tensor
            Fake images from domain B, G_A(A)
        fake_B_pool : ImagePool
            buffer of history of generated images by generator A

        Returns
        -------
        loss_D_A : torch.Tensor
            total discriminator A loss
        """

        fake_B = fake_B_pool.query(fake_B)
        loss_D_A = self.forward_D(netD_A, real_B, fake_B)
        return loss_D_A


    def forward_D_B(self, netD_B, real_A, fake_A, fake_A_pool):
        """
        Computes GAN loss for discriminator B

        Parameters
        ----------
        netD_B : torch.device
            Domain B discriminator network, netD_B: G_A(A) vs. B
        real_A : torch.Tensor
            Real images from domain A
        fake_A : torch.Tensor
            Fake images from domain A, G_B(B)
        fake_A_pool : ImagePool
            buffer of history of generated images by generator B

        Returns
        -------
        loss_D_B : torch.Tensor
            total discriminator B loss
        """

        fake_A = fake_A_pool.query(fake_A)
        loss_D_B = self.forward_D(netD_B, real_A, fake_A)

        return loss_D_B


    def forward(self, generator, **kwargs):
        """Compute either discriminator loss or generator loss"""
        if generator:
            return self.forward_G(**kwargs)
        else:
            return self.forward_D(**kwargs)



class GANLoss(BaseLoss)
    """Generic GAN Loss"""

    def __init__(self, GAN_criterion, target_real_label=1.0,
                 target_fake_label=0.0, weight=None, reduction='mean',
                 **loss_kwargs):
        """
        Parameters
        ----------
        GAN_criterion : str
            The type of GAN criterion, e.g., vanilla, lsgan, wgangp
        target_real_label : bool
           label for a real image
        target_fake_label : bool
           label for a fake image
        weight : torch.Tensor (optional)
            A manual rescaling weight given to the loss of each batch element.
            If given, has to be a Tensor of size nbatch.
        reduction : str (optional)
            Specifies the reduction to apply to the output: 'none' | 'mean' |
            'sum'. 'none': no reduction will be applied, 'mean': the sum of the
            output will be divided by the number of elements in the output,
            'sum': the output will be summed. Note: size_average and reduce are
            in the process of being deprecated, and in the meantime, specifying
            either of those two args will override reduction. Default: 'mean'
        loss_kwargs : dict
            Loss parameters
        """

        super(GANLoss, self).__init__(weight=weight, reduction=reduction)

        self.GANCriterion = get_GAN_criterion(GAN_criterion, **loss_kwargs)

        # Buffers are parameters that should be saved and restored in the
        # state_dict but not trained by the optimizer
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))


    def get_target_tensor(self, prediction: torch.Tensor,
                          target_is_real: bool): -> torch.Tensor

        """
        Creates label tensors with the same size as the input.

        Parameters
        ----------
        prediction : torch.Tensor
            Tipically the prediction from a discriminator
        target_is_real : bool
            If True, the label is for real images, else fake images

        Returns
        -------
        torch.Tensor
            A label tensor filled with ground truth label, and with the same
            size as the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)


    @abstractmethod
    def forward_G(self):
        """Compute Generator(s) loss(es)"""


    @abstractmethod
    def forward_D(self):
        """Compute Discriminator(s) loss(es)"""

    @abstractmethod
    def forward(self, generator):
        """Compute either Discriminator(s) loss(es) or Generator(s) loss(es)"""



def get_GAN_criterion(GAN_criterion: str, **loss_kwargs): -> nn.Loss
    """
    Construct the type GAN criterion based on the supplied name

    Parameters
    ----------
    GAN_criterion : str
        The type of GAN criterion
        One of: vanilla | lsgan | wgangp
    loss_kwargs : dict
        Loss parameters
    """

    GAN_criterions = {
        "vanilla": nn.BCEWithLogitsLoss(**kwargs),
        "lsgan": nn.MSELoss(**kwargs),
        "wgangp": WGANGPLoss(**kwargs)
    }

    if GAN_criterion in GAN_criterions:
        return GAN_criterions[GAN_criterion]
    else:
        msg = (f"GAN criterion name [{GAN_criterion}] "
               f"is not recognized")
        raise NotImplementedError(msg)
