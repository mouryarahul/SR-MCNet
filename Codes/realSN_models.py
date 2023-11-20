import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd as autograd

from jacobian import jac_loss_estimate, power_method
from solvers import conjugate_gradient, anderson_2

from conv_sn_chen import conv_spectral_norm
from bn_sn_chen import bn_spectral_norm

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        torch.nn.init.constant_(m.bias.data, 0.0)

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, lip=1.0, bias=False, bn=True, act='ReLU', skip=False):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        if lip > 0.0:
            sigmas = [pow(lip, 1.0/num_of_layers) for _ in range(num_of_layers)]
        else:
            sigmas = [0.0 for _ in range(num_of_layers)]

        if act == 'CELU':
            print("Selecting CELU activation")
        else:
             print("Selecting RELU activation")

        def conv_layer(cin, cout, sigma):
            conv = nn.Conv2d(in_channels=cin,
                             out_channels=cout,
                             kernel_size=kernel_size,
                             padding=padding,
                             bias=bias)
            if sigma > 0.0:
                return conv_spectral_norm(conv, sigma=sigma)
            else:
                return conv

        def bn_layer(n_features, sigma=1.0):
            bn = nn.BatchNorm2d(n_features)
            if sigma > 0.0:
                return bn_spectral_norm(bn, sigma=sigma)
            else:
                return bn

        layers = []
        layers.append(conv_layer(channels, features, sigmas[0]))
        if act == 'CELU':
            layers.append(nn.CELU(inplace=True))
        else:
             layers.append(nn.ReLU(inplace=True))
        
        # print("conv_1 with SN {}".format(sigmas[0]))

        for i in range(1, num_of_layers-1):
            layers.append(conv_layer(features, features, sigmas[i])) # conv layer
            # print("conv_{} with SN {}".format(i+1, sigmas[i]))
            if bn:
                # print("bn_{}".format(i+1))
                layers.append(bn_layer(features, 0.0)) # bn layer
            if act == 'CELU':
                layers.append(nn.CELU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))

        layers.append(conv_layer(features, channels, sigmas[-1]))
        # print("conv_{} with SN {}".format(num_of_layers, sigmas[-1]))
        self.dncnn = nn.Sequential(*layers)
        self.dncnn.apply(weights_init_kaiming)
        self.skip = skip
        # print("skip = ",self.skip)
        
    def forward(self, x):
        noise = self.dncnn(x)
        if self.skip:
            out = x - noise
        else:
            out = noise
        return out

class FixedPointFuncCase2(nn.Module):
    def __init__(self, dncnn, interp_scale, sigma, epsilon, alpha=1.0, rho=1.0):
        super(FixedPointFuncCase2, self).__init__()
        self.dncnn          = dncnn
        self.interp_scale   = interp_scale
        self.sigma          = sigma
        self.epsilon        = epsilon
        self.rho            = rho
        self.alpha          = Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)

        self.A              = lambda x: F.interpolate(x, size=None, scale_factor=(1/self.interp_scale), align_corners=False, antialias=True, mode='bicubic')
        self.AT             = lambda x: F.interpolate(x, size=None, scale_factor=self.interp_scale, align_corners=False, antialias=True, mode='bicubic')
        self.LHS            = lambda x: self.rho * self.AT(self.A(x)) + (self.alpha + self.rho) * x
        
        
    def forward(self, z, w, b, new_batch=False):
        bsz, C, H, W    = w.shape
        bsz, C, P, Q    = b.shape

        x, lam1, lam2   = torch.split(z, C*H*W, dim=-1)
        x               = x.view(bsz, C, H, W)
        lam1            = lam1.view(bsz, C, H, W)
        lam2            = lam2.view(bsz, C, P, Q)

        # Denoising step: variable u-update
        y               = x + lam1
        y_min, y_max    = y.min(), y.max()
        y               = (y - y_min) / (y_max - y_min)
        scale_range     = 1.0 + self.sigma
        scale_shift     = (1 - scale_range) / 2.0
        y               = y * scale_range + scale_shift
        u               = y - self.dncnn(y)
        u               = (u - scale_shift) / scale_range
        u               = u * (y_max - y_min) + y_min

        # Projection on Ball: || x - t ||_2 <= epsilon
        y               = (self.A(x) - b + lam2)
        v               = (y*torch.minimum(self.epsilon/torch.linalg.norm(y.view(bsz,C,-1,1), dim=2, keepdim=True), torch.tensor(1.0)))

        # x-update
        rhs             = self.alpha * w + self.rho * (u - lam1) + self.rho * self.AT(v + b - lam2)
        x               = conjugate_gradient(x, self.LHS, rhs)

        # Dual-variable updates
        lam1            = lam1 + x - u
        lam2            = lam2 + self.A(x) - v - b

        z               = torch.cat((x.view(bsz, -1), lam1.view(bsz, -1), lam2.view(bsz, -1)), dim=-1)
        return z


class DEQNet(nn.Module):
    def __init__(self, f, solver, case, **kwargs):
        super(DEQNet, self).__init__()
        self.f                  = f
        self.solver             = solver
        self.case               = case
        self.hook               = None
        self.max_iter_forward   = kwargs.get('max_iter_forward')
        self.min_iter_forward   = kwargs.get('min_iter_forward')
        self.max_iter_backward  = kwargs.get('max_iter_backward')
        self.min_iter_backward  = kwargs.get('min_iter_backward')
        self.tol                = kwargs.get('tol')
        self.beta               = kwargs.get('beta')
        self.verbose            = kwargs.get('verbose')
        
    def forward(self, w, b, compute_jac_loss=False, spectral_radius_mode=False):
        bsz, C, H, W        = w.shape
        jac_loss            = torch.tensor(0.0).to(w)
        sradius             = torch.zeros(bsz, 1).to(w)

        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            x = torch.clone(w)
            lam1    = torch.zeros_like(w)
            lam2    = torch.zeros_like(b)
            z0      = torch.cat((x.view(bsz, -1), lam1.view(bsz, -1), lam2.view(bsz, -1)), dim=-1)
            
                
            # compute forward pass and re-engage autograd tape
            z = self.f(z0, w, b, new_batch=True)  # First call to precalculate some terms in the Fixed-point iterations
            z, forward_res = self.solver(lambda z : self.f(z, w, b), z, 
                                        min_iter=self.min_iter_forward, 
                                        max_iter=self.max_iter_forward, 
                                        tol=self.tol, beta=self.beta, 
                                        verbose=self.verbose)
        z_new = self.f(z, w, b)

        if (not self.training) and spectral_radius_mode:
            with torch.enable_grad():
                z0 = z.clone().detach().requires_grad_()
                f0 = self.f(z0, w, b)
            _, sradius = power_method(f0, z0, n_iters=150)

        # set up Jacobian vector product (without additional forward calls)
        if self.training:
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(z0, w, b)
           
            if compute_jac_loss:
                jac_loss = jac_loss_estimate(f0, z0, vecs=1)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove() # To avoid infinite loop
                    if w.device.type == 'cuda': 
                        torch.cuda.synchronize()
                new_grad, backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, 
                                                    grad, min_iter=self.min_iter_backward, max_iter=self.max_iter_backward, 
                                                    tol=self.tol, beta=self.beta, verbose=self.verbose)
                return new_grad
            self.hook = z_new.register_hook(backward_hook)

        new_x, new_lam1, new_lam2 = torch.split(z_new, C*H*W, dim=-1)

        return new_x.view(bsz, C, H, W), jac_loss.view(-1,1), sradius.view(-1,1)
