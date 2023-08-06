import torch
from functools import partial


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



# Linear MMD (DDC)
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


# MK-MMD (DAN)
# extracted from:
# https://github.com/easezyc/deep-transfer-learning

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),
                                        int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)),
                                        int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) 
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mk_mmd_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
#         loss += kernels[s1, s2] + kernels[t1, t2]
#         loss -= kernels[s1, t2] + kernels[s2, t1]
        loss_ = (kernels[s1, s2] + kernels[t1, t2]) - (kernels[s1, t2] + kernels[s2, t1])
        loss += torch.clamp(loss_, min=0.0)
    return loss / float(batch_size)

# this version is similar to https://github.com/CPJKU/da
# which where adapted from https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
def mk_mmd_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, 
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


# RBF MMD (DSN)
# extracted from:
# https://github.com/HKUST-KnowComp/FisherDA

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