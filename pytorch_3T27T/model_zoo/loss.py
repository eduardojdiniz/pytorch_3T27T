#!/usr/bin/env python
# coding=utf-8

from typing import Optional
from ABC import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from pytorch_3T27T.base import BaseLoss
from pytorch_3T27T.utils import ImagePool


Tensor = torch.Tensor
Module = nn.Module


__all__ = ['WGANGPLoss', 'Pix2PixLoss', 'CycleGANLoss']


def get_GAN_criterion(GAN_criterion: str, **loss_kwargs) -> Module:
    """
    Construct the type GAN criterion based on the supplied name

    Parameters
    ----------
    GAN_criterion : str
        The type of GAN criterion.
        One of: vanilla | lsgan | wgangp.
    loss_kwargs : dict
        Loss parameters.
    """

    GAN_criteria = {
        "vanilla": nn.BCEWithLogitsLoss(**loss_kwargs),
        "lsgan": nn.MSELoss(**loss_kwargs),
        "wgangp": WGANGPLoss(**loss_kwargs)
    }

    if GAN_criterion in GAN_criteria:
        return GAN_criteria[GAN_criterion]
    else:
        msg = (f"GAN criterion name [{GAN_criterion}] "
               f"is not recognized")
        raise NotImplementedError(msg)


class GANLoss(BaseLoss, Metaclass=ABCMeta):
    """Generic GAN Loss"""

    def __init__(self, GAN_criterion: str, target_real_label: float = 1.0,
                 target_fake_label: float = 0.0,
                 weight: Optional[Tensor] = None,
                 reduction: Optional[str] = 'mean', **loss_kwargs) -> None:
        """
        Parameters
        ----------
        GAN_criterion : str
            The type of GAN criterion, e.g., vanilla, lsgan, wgangp.
        target_real_label : float
           label for a real image.
        target_fake_label : float
           label for a fake image.
        weight : Tensor (optional)
            A manual rescaling weight given to the loss of each batch element.
            If given, has to be a Tensor of size nbatch.
        reduction : str (optional)
            Specifies the reduction to apply to the output: 'none' | 'mean' |
            'sum'. 'none': no reduction will be applied, 'mean': the sum of the
            output will be divided by the number of elements in the output,
            'sum': the output will be summed. Note: size_average and reduce are
            in the process of being deprecated, and in the meantime, specifying
            either of those two args will override reduction. Default: 'mean'.
        loss_kwargs : dict
            Loss parameters.
        """

        super().__init__(weight=weight, reduction=reduction)

        self.GANCriterion = get_GAN_criterion(GAN_criterion, **loss_kwargs)

        # Buffers are parameters that should be saved and restored in the
        # state_dict but not trained by the optimizer
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    def get_target_tensor(self, prediction: Tensor,
                          target_is_real: bool) -> Tensor:
        """
        Creates label tensors with the same size as the input.

        Parameters
        ----------
        prediction : Tensor
            Tipically the prediction from a discriminator.
        target_is_real : bool
            If True, the label is for real images, else fake images.

        Returns
        -------
        _ : Tensor
            A label tensor filled with ground truth label, and with the same
            size as the input.
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    @abstractmethod
    def forward_G(self):
        """Compute Generator(s) loss(es)"""

    @abstractmethod
    def forward_D(self):
        """Compute Discriminator(s) loss(es)"""

    @abstractmethod
    def forward(self):
        """Compute either Discriminator(s) loss(es) or Generator(s) loss(es)"""


class WGANGPLoss(BaseLoss):
    """
    Implement the gradient penalty loss.
    WGAN-GP paper https://arxiv.org/abs/1704.00028
    """

    def __init__(self, data_type: str = "mixed", constant: float = 1.0,
                 lambda_gp: float = 10.0, weight: Optional[Tensor] = None,
                 reduction: Optional[str] = 'mean') -> None:
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
        weight : Tensor (optional)
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

        super().__init__(weight=weight, reduction=reduction)

        self.data_type = data_type
        self.constant = constant
        self.lambda_gp = lambda_gp

    def forward(self, netD: Module, real: Tensor, fake: Tensor) -> Tensor:
        """
        Computes the gradient penalty loss

        Parameters
        ----------
        netD : Module
            Discriminator network
        real : Tensor
            Real images
        fake : Tensor
            Fake images

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
            loss = ((gradients + 1e-16).norm(2, dim=1) - self.constant) ** 2
            loss = loss.mean() * self.lambda_gp
            return loss
        else:
            return 0.0

    def get_output_from_type(self, real: Tensor, fake: Tensor) -> Tensor:
        """
        Get output parameter for forward passed based on data_type

        Parameters
        ----------
        real : Tensor
            Real images
        fake : Tensor
            Fake images

        Returns
        -------
        output : Tensor
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

    def __init__(self, lambda_L1: Tensor, GAN_criterion: str = 'vanilla',
                 target_real_label: float = 1.0,
                 target_fake_label: float = 0.0,
                 weight: Optional[Tensor] = None,
                 reduction: Optional[str] = 'mean') -> None:
        """
        Parameters
        ----------
        lambda_L1 : Tensor
            The L1 regularization weight
        GAN_criterion : str
            The type of GAN criterion. Default: vanilla
        target_real_label : float
            label for a real image. Default: 1.0
        target_fake_label : float
            label for a fake image. Default: 0.0
        weight : Tensor (optional)
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

    def forward_G(self, netD: Module, real_A: Tensor, real_B: Tensor,
                  fake_B: Tensor) -> Tensor:
        """
        Computes GAN and L1 loss for the generator

        Parameters
        ----------
        netD : Module
            Discriminator network.
        real_A : Tensor
            Real images from domain A.
        real_B : Tensor
            Real images from domain B.
        fake_B : Tensor
            Fake images from domain B.

        Returns
        -------
        loss_G : Tensor
            Total generator loss.
        """

        # First, Generator should fake the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB)
        target_fake = self.get_target_tensor(pred_fake, True)
        loss_G_GAN = self.GANCriterion(pred_fake, target_fake)

        # Second G(A) = B
        loss_G_L1 = self.L1Criterion(fake_B, real_B)

        # Combine losses
        loss_G = loss_G_GAN + self.lambda_L1 * loss_G_L1

        return loss_G

    def forward_D(self, netD: Module, real_A: Tensor, real_B: Tensor,
                  fake_B: Tensor) -> Tensor:
        """
        Computes the discriminator loss

        Parameters
        ----------
        netD : Module
            Discriminator network.
        real_A : Tensor
            Real images from domain A.
        real_B : Tensor
            Real images from domain B.
        fake_B : Tensor
            Fake images from domain B.

        Returns
        -------
        loss_D : Tensor
            Total discriminator loss.
        """

        # Fake

        # We use conditional GANs; we need to feed both input and output to the
        # discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)

        # Stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake_AB.detach())
        target_fake = self.get_target_tensor(pred_fake, False)
        loss_D_fake = self.GANCriterion(pred_fake, target_fake)

        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB)
        target_real = self.get_target_tensor(pred_real, True)
        loss_D_real = self.GANCriterion(pred_real, target_real)

        # Combine losses
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def forward(self, netD: Module, real_A: Tensor, real_B: Tensor,
                fake_B: Tensor, generator: bool = True) -> Tensor:
        """Compute either discriminator loss or generator loss"""
        if generator:
            return self.forward_G(netD, real_A, real_B, fake_B)
        else:
            return self.forward_D(netD, real_A, real_B, fake_B)


class CycleGANLoss(GANLoss):
    """Define the Cycle GAN Loss for all the discriminators and generators"""

    def __init__(self, lambda_idt: Tensor, lambda_A: Tensor, lambda_B: Tensor,
                 GAN_criterion: str = 'vanilla',
                 target_real_label: float = 1.0,
                 target_fake_label: float = 0.0,
                 weight: Optional[Tensor] = None,
                 reduction: Optional[str] = 'mean') -> None:
        """
        Parameters
        ----------
        lambda_idt : Tensor
            The Identity loss weight
        lambda_A : Tensor
            The Cycle real_A -> fake_B -> rec_A loss weight
        lambda_B : Tensor
            The Cycle real_B -> fake_A -> rec_B loss weight
        GAN_criterion : str
            The type of GAN criterion. Default: vanilla
        target_real_label : float
            label for a real image. Default: 1.0
        target_fake_label : float
            label for a fake image. Default: 0.0
        weight : Tensor (optional)
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
        self.lambda_idt = lambda_idt

    def forward_G(self, netD_A: Module, netD_B: Module, real_A: Tensor,
                  real_B: Tensor, fake_A: Tensor, fake_B: Tensor,
                  rec_A: Tensor, rec_B: Tensor, idt_A: Tensor,
                  idt_B: Tensor) -> Tensor:
        """
        Computes GAN, Cycle and Identity losses for the generators

        Parameters
        ----------
        netD_A : Module
            Domain A discriminator network, netD_A: G_B(B) vs. A.
        netD_B : Module
            Domain B discriminator network, netD_B: G_A(A) vs. B.
        real_A : Tensor
            Real images from domain A.
        real_B : Tensor
            Real images from domain B.
        fake_A : Tensor
            Fake images from domain A, G_B(B).
        fake_B : Tensor
            Fake images from domain B, G_A(A).
        rec_A : Tensor
            Reconstructed images from domain A, G_B(G_A(A)).
        rec_B : Tensor
            Reconstructed images from domain B, G_A(G_B(B)).
        idt_A : Tensor
            Identity images from domain B, G_A(B).
        idt_B : Tensor
            Identity images from domain A, G_B(A).

        Returns
        -------
        loss_G : Tensor
            total generation loss.
        """

        # First, Generators should fake the discriminators
        # GAN loss D_A(G_A(A))
        pred_fake_B = netD_A(fake_B)
        target_true = self.get_target_tensor(pred_fake_B, True)
        loss_GAN_G_A = self.GANCriterion(pred_fake_B, target_true)
        # GAN loss D_B(G_B(B))
        pred_fake_A = netD_B(fake_A)
        target_true = self.get_target_tensor(pred_fake_A, True)
        loss_GAN_G_B = self.GANCriterion(pred_fake_A, target_true)

        # Second, we ensure cycle consistency
        # Forward cycle loss || G_B(G_A(A)) - A ||
        loss_cycle_A = self.CycleCriterion(rec_A, real_A)
        # Backward cycle loss || G_A(G_B(B)) - B ||
        loss_cycle_B = self.CycleCriterion(rec_B, real_B)

        # Third, we ensure identity mapping
        if self.lambda_idt > 0:
            # G_A should be identity if real_B is fed: "||G_A(B) - B||"
            loss_idt_A = self.IdtCriterion(self.idt_A, real_B)
            # G_B should be identity if real_A is fed: "||G_B(A) - A||"
            loss_idt_B = self.IdtCriterion(self.idt_B, real_A)
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # Combine losses
        loss_G = 0
        loss_G += loss_idt_A * self.lambda_B * self.lambda_idt
        loss_G += loss_idt_B * self.lambda_A * self.lambda_idt
        loss_G += loss_GAN_G_A
        loss_G += loss_GAN_G_B
        loss_G += loss_cycle_A * self.lambda_A
        loss_G += loss_cycle_B * self.lambda_B
        return loss_G

    def forward_D(self, netD: Module, real: Tensor, fake: Tensor):
        """
        Computes the discriminators losses

        Parameters
        ----------
        netD : Module
            Discriminator network, netD: G(real) vs. real.
        real : Tensor
            Real images.
        fake : Tensor
            Fake images, G(real).

        Returns
        -------
        loss_D : Tensor
            total discriminator loss.
        """

        # Real
        pred_real = netD(real)
        target_real = self.get_target_tensor(pred_real, True)
        loss_D_real = self.GANCriterion(pred_real, target_real)

        # Fake
        # Stop backprop to the generator by detaching fake
        pred_fake = netD(fake.detach())
        target_fake = self.get_target_tensor(pred_fake, False)
        loss_D_fake = self.GANCriterion(pred_fake, target_fake)

        # Combine losses
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def forward_D_A(self, netD_A: Module, real_B: Tensor, fake_B: Tensor,
                    fake_B_pool: ImagePool) -> Tensor:
        """
        Computes GAN loss for discriminator A

        Parameters
        ----------
        netD_B : Module
            Domain B discriminator network, netD_B: G_A(A) vs. B.
        real_B : Tensor
            Real images from domain A.
        fake_A : Tensor
            Fake images from domain A, G_B(B).
        fake_B : Tensor
            Fake images from domain B, G_A(A).
        fake_B_pool : ImagePool
            buffer of history of generated images by generator A.

        Returns
        -------
        loss_D_A : Tensor
            total discriminator A loss.
        """

        fake_B = fake_B_pool.query(fake_B)
        loss_D_A = self.forward_D(netD_A, real_B, fake_B)
        return loss_D_A

    def forward_D_B(self, netD_B: Module, real_A: Tensor, fake_A: Tensor,
                    fake_A_pool: ImagePool) -> Tensor:
        """
        Computes GAN loss for discriminator B

        Parameters
        ----------
        netD_B : Module
            Domain B discriminator network, netD_B: G_A(A) vs. B.
        real_A : Tensor
            Real images from domain A.
        fake_A : Tensor
            Fake images from domain A, G_B(B).
        fake_A_pool : ImagePool
            buffer of history of generated images by generator B.

        Returns
        -------
        loss_D_B : Tensor
            total discriminator B loss.
        """

        fake_A = fake_A_pool.query(fake_A)
        loss_D_B = self.forward_D(netD_B, real_A, fake_A)

        return loss_D_B

    def forward(self, generator: bool = True, **kwargs):
        """Compute either discriminator loss or generator loss"""
        if generator:
            return self.forward_G(**kwargs)
        else:
            return self.forward_D(**kwargs)
