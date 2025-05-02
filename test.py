# import os
# if os.path.exists("rna_3d_model.pth"):
#     print("yes")
# else:
#     print("no")

import torch

x = torch.tensor([[[123,12,1],[2,1,3]],[[123,12,1],[2,1,3]]])
y = torch.tensor([1,2,3])
# y = torch.nn.functional.sigmoid(y)
# y = y.unsqueeze(-1).expand_as(x)  # [B, T, D]
print(x[:,:,1])
# print(x*y)