import torch
a = torch.Tensor([[1,2,3],[4,5,6]])
b = torch.ones(2,3)
print(torch.stack((a,b)).shape)
d = torch.sum(torch.stack((a,b), axis=1))
print(d)
