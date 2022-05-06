
import numpy as np

import torch
import torch.nn as nn

class IdentityTrans(nn.Module): 
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(IdentityTrans, self).__init__()

        self.input_shape = input_shape

    @torch.no_grad()
    def update(self, input_vector, z_idxs):
        pass

    def normalize(self, input_vector):
        
        return input_vector

    def denormalize(self, input_vector):
        
        return input_vector


class ValueNorm(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, args, input_shape, norm_axes=1, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = 1 - 1e-2
        self.max_z = args.max_z
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)
        
        self.running_mean_z = [
            nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv) \
            for _ in range(self.max_z)
        ]
        self.debiasing_term_z = [
            nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv) \
            for _ in range(self.max_z)
        ]
        
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()
        for z in range(self.max_z):
            self.running_mean_z[z].zero_()
            self.debiasing_term_z[z].zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def get_z_mean(self):
        results = []
        for z in range(self.max_z):
            z_mean = self.running_mean_z[z] / self.debiasing_term_z[z].clamp(min=self.epsilon)
            results.append(z_mean)
        results = torch.cat(results)
        return results

    @torch.no_grad()
    def update(self, input_vector, z_idxs):
        
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)

        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        # update z_para
        for z in range(self.max_z):
            input_z_vector = input_vector[z_idxs==z]
            batch_z_mean = input_z_vector.mean()
            if input_z_vector.shape[0] > 0:
                self.running_mean_z[z].mul_(self.beta).add_(batch_z_mean * (1.0 - self.beta))
                self.debiasing_term_z[z].mul_(self.beta).add_(1.0 * (1.0 - self.beta))

    def normalize(self, input_vector):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()
        
        return out
