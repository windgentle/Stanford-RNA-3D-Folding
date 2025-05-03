# import os
# if os.path.exists("rna_3d_model.pth"):
#     print("yes")
# else:
#     print("no")

import torch

# x = torch.tensor([[[123,12,1],[2,1,3]],[[123,12,1],[2,1,3]]])
# y = torch.tensor([1,2,3])
# # y = torch.nn.functional.sigmoid(y)
# # y = y.unsqueeze(-1).expand_as(x)  # [B, T, D]
# print(x[:,:,1])
# # print(x*y)
B, T, K, D = 2, 4, 3, 3 
coords1 = torch.rand(B, T, K, D)  # 第一个 [B, T, K, 3] 张量
coords2 = torch.rand(B, T, K, D)  # 第一个 [B, T, K, 3] 张量

weights = torch.rand(B, T, K)  # 第二个 [B, T, K, 3] 张量

print(f"coords1 shape: {coords1.shape}")
print(f"coords1: {coords1}")
print(f"coords2 shape: {weights.shape}")
print(f"coords2: {weights}")

def weighted_avg(coords, weights, eps=1e-8):
    """
    coords: [B, T, K, 3]
    weights: [B, T, K]
    returns: [B, T, 3]
    """
    weights = weights.to(coords.dtype)  # 类型统一，防止 float64 × float32 报错
    weights = weights.unsqueeze(-1)  # [B, T, K, 1]
    
    weighted_sum = torch.sum(coords * weights, dim=2)  # [B, T, 3]
    total_weight = torch.sum(weights, dim=2)  # [B, T, 1] after unsqueeze
    
    # 防止除以 0（当所有 K 的 confidence 为 0）
    total_weight = total_weight.clamp(min=eps)  # [B, T, 1]
    
    avg = weighted_sum / total_weight  # [B, T, 3]
    return avg

def kabsch_align(P, Q):
    """
    P, Q: [B, T, 3] - predicted & target coordinates
    returns: aligned P (rigid-body aligned to Q)
    """
    P_mean = P.mean(dim=1, keepdim=True)
    Q_mean = Q.mean(dim=1, keepdim=True)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    H = torch.matmul(P_centered.transpose(1, 2), Q_centered)  # [B, 3, 3]
    U, S, Vh = torch.linalg.svd(H)
    d = torch.linalg.det(torch.matmul(Vh, U.transpose(1, 2)))
    D = torch.diag_embed(torch.ones_like(d).repeat(3, 1).T)
    D[:, 2, 2] = d
    R = torch.matmul(Vh, torch.matmul(D, U.transpose(1, 2)))  # [B, 3, 3]

    aligned_P = torch.matmul(P_centered, R.unsqueeze(1)) + Q_mean  # [B, T, 3]
    return aligned_P

def confidence_rmsd_loss(pred, target, confidence):
    """
    pred, target: [B, T, K, 3]
    confidence: [B, T, K]
    returns: scalar RMSD loss
    """
    print(f"pred shape: {pred.shape}")
    print(f"target shape: {target.shape}")
    print(f"confidence shape: {confidence.shape}")
    # Step 1: confidence-weighted average across K
    pred_avg = weighted_avg(pred, confidence)      # [B, T, 3]
    target_avg =  weighted_avg(target, confidence)  # [B, T, 3]

    # Step 2: rigid alignment
    pred_aligned = kabsch_align(pred_avg, target_avg)  # [B, T, 3]

    # Step 3: RMSD
    diff = pred_aligned - target_avg
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=1))  # [B]
    return rmsd.mean()

# re = weighted_avg(coords1,weights)

print(confidence_rmsd_loss(coords1,coords2,weights))