from datasets.augmentations import ToOneHot
import torch
x = torch.tensor([[0, 1, 1, 2, 0, 0, 0, 0, 0]]).int()
x = x.view(1, 1, 9)
print(x)
print(x.shape)
onehot = ToOneHot(10)
y = onehot(x)
print(y)
print(y.shape)