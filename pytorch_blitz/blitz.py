import torch

x = torch.empty(5,3)
x = torch.zeros(5,3, dtype=torch.short)
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)
x = x.new_zeros(5, 3, dtype=torch.double)
x = torch.randn_like(x, dtype=torch.float)
x = x.size() # torch.Size is tuple
x = torch.rand(5, 3)
y = torch.rand(5, 3)
z = x + y
x = torch.add(x, y)
x = torch.zeros(5, 3)
torch.add(y, z, out=x)
torch.rand(5,3)
x.add_(x)

x = torch.randn(4, 4)
x = x.view(-1, 2, 2)

x = torch.randn(1)

a = torch.ones(5)
b = a.numpy()

#if torch.cuda.is_available():
#    device = torch.device("cuda") # CUDA device object
#    y = torch.ones_like(x, device=device) # create tensor on GPU
#    x = x.to(device)
#    z = x + y

x = torch.ones(2, 2, requires_grad=True)

print(x)



