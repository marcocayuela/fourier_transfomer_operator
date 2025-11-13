import torch.nn as nn
import numpy as np

class LiftingLayer(nn.Module):

    """Lifting Layer to map input data to higher dimensional space.
    Args:
        input_dim (int): dimension of the input data;
        output_dim (int): dimension of the output data;
        device (str, optional): device to run the model on. Defaults to "cpu".
    Returns:
        nn.Module: Lifting Layer model.
    """    

    def __init__(self, input_dim, output_dim, device="cpu"):
        super(LiftingLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.device = device

    def forward(self, x):
        return self.linear(x)
    


class FourierBining():
    """Fourier Bining Layer to separate Fourier modes into different bins.
    Args:
        modes_separation (list): list of integers defining the separation of modes;
        domain_size (int): size of the domain;
        device (str, optional): device to run the model on. Defaults to "cpu".
    Returns:
        FourierBining class: Fourier Bining Layer model.
    """

    def __init__(self, modes_separation, n_dim, domain_size=None, norm="L2", device="cpu"):
        super(FourierBining, self).__init__()
        self.modes_separation = modes_separation
        self.freq_max = self.modes_separation[-1]
        self.n_dim = n_dim
        self.domain_size = n_dim*[1.] if domain_size is None else domain_size
        self.norm = norm
        self.device = device

    def magnitude_spectrum(self):
        k_axes = []
        for L in self.domain_size:
            N = self.freq_max*2
            dx = L/N
            k = np.fft.fftfreq(N, d=dx)
            k_axes.append(k)

        self.k_mesh = np.meshgrid(*k_axes, indexing='ij')
        self.k_magnitude = self.compute_freq_magnitude(self.k_mesh) 
        self.rk_magnitude = self.k_magnitude[...,:self.freq_max+1]

    def compute_freq_magnitude(self, k_mesh):
        if self.norm=="L2":
            k_magnitude = np.sqrt(np.sum([k**2 for k in k_mesh], axis=0)) 
        elif self.norm=="L1":
            k_magnitude = np.sum([np.abs(k) for k in k_mesh], axis=0) 
        elif self.norm=="max":
            k_magnitude = np.max([np.abs(k) for k in k_mesh], axis=0)
        else:
            raise NotImplementedError(f"Norm {self.norm} not implemented.")
        return k_magnitude

    def create_masks(self):
        self.masks = []
        for i in range(len(self.modes_separation)):
            if i==0:
                k_min = 0
            else:
                k_min = self.modes_separation[i-1]
            k_max = self.modes_separation[i]
            mask = np.logical_and(self.rk_magnitude>=k_min, self.rk_magnitude<k_max)
            self.masks.append(mask)
