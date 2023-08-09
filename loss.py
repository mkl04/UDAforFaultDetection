import torch
from functools import partial

# Loss for DexiNed
# extracted from: https://github.com/xavysp/DexiNed

def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float()
    num_negative = torch.sum((mask <= 0.0).float()).float()
    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = torch.sum(cost.float().mean((1, 2, 3)))
    return l_weight*cost

# RBF MMD (DSN)
# extracted from: https://github.com/HKUST-KnowComp/FisherDA

def compute_pairwise_distances(x, y):
    if not x.dim() == y.dim() == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size(1) != y.size(1):
        raise ValueError('The number of features should be the same.')

    norm = lambda x: torch.sum(torch.pow(x, 2), 1)
    return torch.transpose(norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1)

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    # print('dist shape={}'.format(dist.size()))
    s = torch.matmul(beta, dist.contiguous().view(1, -1))
    return torch.sum(torch.exp(-s), 0).view(*dist.size())

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    # We do not allow the loss to become negative.
    cost = torch.clamp(cost, min=0.0)
    return cost

def mmd_dsn(hs, ht):
    '''maximum mean discrepancy, a combination of multiple kernels
    '''
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
              10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix,
                              sigmas=torch.Tensor(sigmas).float().cuda())
    loss_value = maximum_mean_discrepancy(hs, ht, kernel=gaussian_kernel)
    return torch.clamp(loss_value, min=1e-4)