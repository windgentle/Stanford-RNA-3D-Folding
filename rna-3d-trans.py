import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import random
import string
from torchtune.modules import RotaryPositionalEmbeddings
from torch.utils.data import Dataset, DataLoader,random_split
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

#const
# device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 5 #输出数量

#数据标准化
'''
x 坐标均值: 172.22412267209705, 方差: 111.78219610710121
y 坐标均值: 167.04462135982504, 方差: 119.23954077924417
z 坐标均值: 166.03082937929065, 方差: 121.29372992890671
'''
COORDS_MEAN = torch.tensor([172.2241, 167.0446, 166.0308], device=device)
COORDS_STD = torch.tensor([111.7822, 119.2395, 121.2937], device=device)

# 固定 vocab（包含 <pad>）
VOCAB = ['<pad>'] + list("ACGUX-")
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

# RNA_PARA_FILE = "/kaggle/input/rna-llm-parameter-file/"
RNA_PARA_FILE = ""
RNA_TRANS_MODEL_FILE = f"{RNA_PARA_FILE}rna_transformer_model.pth"
RNA_RNN_MODEL_FILE = f"{RNA_PARA_FILE}rna_3d_model.pth"
RNA_REPR_CACHE_FILE = f"{RNA_PARA_FILE}string_repr_cache.pt"
RNA_PERP_CACHE_VAL_FILE = f"{RNA_PARA_FILE}string_repr_cache_val.pt"

DATAPATH="stanford-rna-3d-folding"
TRAIN_SEQ_FILE_PATH = f"{DATAPATH}/train_sequences_cleaned.csv"
TRAIN_LABEL_FILE_PATH = f"{DATAPATH}/train_labels_cleaned.csv"
VALI_SEQ_FILE_PATH = f"{DATAPATH}/validation_sequences.csv"
VALI_LABEL_FILE_PATH = f"{DATAPATH}/validation_labels.csv"
TEST_SEQ_FILE_PATH = f"{DATAPATH}/test_sequences.csv"



#序列字符串整体生成向量模型
class TransformerBlockWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, rope):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.rope = rope
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        dim_per_head = d_model // self.attn.num_heads
        q = k = v = x.view(batch_size, seq_len, self.attn.num_heads, dim_per_head)  # 调整形状
        q= self.rope(q)
        k= self.rope(k)
        q = q.view(batch_size, seq_len, d_model)  # 恢复形状
        k = k.view(batch_size, seq_len, d_model)  # 恢复形状
        v = v.view(batch_size, seq_len, d_model)
        attn_out, _ = self.attn(q, k, v)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class SimpleCharTransformerWithRoPE(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbeddings(dim=d_model // nhead, max_seq_len=8192)
        self.layers = nn.ModuleList([
            TransformerBlockWithRoPE(d_model, nhead, self.rope)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
        self.hidden_states = []

    def forward(self, x):
        emb = self.embedding(x)  # [B, T, D]
        h = emb
        self.hidden_states = []
        for layer in self.layers:
            h = layer(h)
            self.hidden_states.append(h.detach())
        return self.output(h)
    
#序列字符串推理实现
def encode_string(s, stoi):
    ids = [stoi[c] for c in s]
    return torch.tensor([ids], dtype=torch.long)  # shape [1, T]

def rna_str_eval(input_str): #torch.Size([1, 10, 512])
    model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
    model.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device,weights_only=True))
    model.eval()  

    # 输入字符串
    input_tensor = encode_string(input_str,stoi).to(device)
    
    # 前向传播
    with torch.no_grad():
        logits = model(input_tensor)
    char_emb = model.embedding(input_tensor)  # [B, T, D]
    last_hidden = model.hidden_states[-1]  # [B, T, D]
    mask = (input_tensor != 0).unsqueeze(-1).float()
    valid_lengths = mask.sum(dim=1)  # [B, 1]
    string_repr = (last_hidden * mask).sum(dim=1) / valid_lengths  # [B, D]
    fused_char_repr = char_emb * string_repr
    # string_repr_expanded = string_repr.unsqueeze(1).expand_as(char_emb)  # [B, T, D]
    # fused_char_repr = char_emb + string_repr_expanded  # [B, T, D]
    return fused_char_repr

#获取rna训练数据
def load_rna_3d_data(seq_file, label_file):
    """
    从Stanford RNA数据集中加载RNA序列和对应3D坐标（不做滑动窗口切分，直接加载完整序列）。    """
    seq_df = pd.read_csv(seq_file)
    label_df = pd.read_csv(label_file)

    data = []
    for _, row in seq_df.iterrows():
        rna_id = row['target_id']
        sequence = row['sequence']
        seq_len = len(sequence)

        # Stanford RNA数据集的label文件ID格式: {rna_id}_{idx}
        coords = label_df[label_df['ID'].str.startswith(rna_id + "_")][['x_1', 'y_1', 'z_1']].values
        #标准化操作
        coords = (coords - COORDS_MEAN.cpu().numpy()) / COORDS_STD.cpu().numpy()

        if seq_len != len(coords):
            print(f"Warning: Sequence length and coordinates length mismatch for RNA ID {rna_id}. Skipping.")
            continue  # 长度不匹配的跳过
        full_ids = torch.tensor([stoi.get(c, 0) for c in sequence], dtype=torch.long)
        ids = [stoi.get(c, 0) for c in sequence]
        x = torch.tensor(ids, dtype=torch.long)
        coord_slice = np.repeat(coords[:, np.newaxis, :], K, axis=1)  # [T, 1, 3] -> [T, K, 3]
        y = torch.tensor(coord_slice, dtype=torch.float)
        data.append((x, y, full_ids))
    return data

class RNACoordsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    xs, ys, full_ids = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0.0)
    full_ids = pad_sequence(full_ids, batch_first=True, padding_value=0)
    return xs, ys, full_ids

        
#使用Transformer模型做预测
class Trans3DPredictor(nn.Module):
    def __init__(self, hidden_dim=512,k=5,num_layers=2,nhead=2):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=hidden_dim // nhead, max_seq_len=8192)   
        self.layers = nn.ModuleList([
            TransformerBlockWithRoPE(hidden_dim, nhead, self.rope)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=0.1)
        self.output_layer = nn.Linear(hidden_dim, 3 * k)
        self.confidence_layer = nn.Linear(hidden_dim, k)  # 输出置信度
        self.k = k
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) 
        x = self.dropout(x)
        out = self.output_layer(x)  # [B, T, 3*K]
        pred_coords = out.view(x.size(0), x.size(1), self.k, 3)  # [B, T, K, 3]
        confidence_out = self.confidence_layer(x)  # [B, T, K]
        confidence = torch.sigmoid(confidence_out)  # 将置信度限制在 [0, 1]
        return pred_coords,confidence


import torch

def weighted_avg(coords, weights, eps=1e-8):
    """
    coords: [B, T, K, 3]
    weights: [B, T, K]
    returns: [B, T, 3]
    """
    weights = weights.unsqueeze(-1)  # [B, T, K, 1]
    weighted_sum = (coords * weights).sum(dim=2)  # [B, T, 3]
    total_weight = weights.sum(dim=2, keepdim=True) + eps  # [B, T, 1]
    return weighted_sum / total_weight  # [B, T, 3]

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
    # Step 1: confidence-weighted average across K
    pred_avg = weighted_avg(pred, confidence)      # [B, T, 3]
    target_avg =  (target, confidence)  # [B, T, 3]

    # Step 2: rigid alignment
    pred_aligned = kabsch_align(pred_avg, target_avg)  # [B, T, 3]

    # Step 3: RMSD
    diff = pred_aligned - target_avg
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=1))  # [B]
    return rmsd.mean()



def weighted_tm_score_loss(pred, target, confidence, alpha=10,L_ref=None):
    """
    加权 TM-score 损失函数，结合置信度。
    
    参数:
    - pred: 模型预测的坐标，形状为 [B, T, K, 3]
    - target: 真实的坐标，形状为 [B, T, K, 3]
    - confidence: 每个候选值的置信度，形状为 [B, T, K]
    - alpha: L2 损失的权重系数
    
    返回:
    - 综合损失值
    """
    B, T, K, _ = pred.shape
    assert pred.shape == target.shape, "预测和目标的形状必须一致"
    assert confidence.shape == (B, T, K), "置信度的形状必须为 [B, T, K]"
    
     # 如果未提供 L_ref，则默认为 target 的长度
    if L_ref is None:
        L_ref = T

    # 计算 d_0
    if L_ref >= 30:
        d_0 = 0.6 * ((L_ref - 0.5) ** 0.5) - 2.5
    else:
        if L_ref < 12:
            d_0 = 0.3
        elif 12 <= L_ref <= 15:
            d_0 = 0.4
        elif 16 <= L_ref <= 19:
            d_0 = 0.5
        elif 20 <= L_ref <= 23:
            d_0 = 0.6
        elif 24 <= L_ref <= 29:
            d_0 = 0.7

    # TM-score 计算
    all_tm_scores = []
    for k in range(K):
        pred_struct = pred[:, :, k, :]     # [B, T, 3]
        target_struct = target[:, :, k, :] # [B, T, 3]

        # Pairwise L2 distance per point
        squared_dists = torch.sum((pred_struct - target_struct)**2, dim=-1) # [B, T]
        dists = torch.sqrt(torch.clamp(squared_dists, min=1e-8))                   # [B, T]

        # 计算 TM-score 分量
        tm_score_components = 1 / (1 + (dists / d_0) ** 2)  # [B, T]

        # 加权 TM-score
        weighted_tm_score = tm_score_components * confidence[:, :, k]  # [B, T]
        tm_scores = weighted_tm_score.mean(dim=1)  # [B]
        all_tm_scores.append(tm_scores)

    # 所有 K 种预测中取最佳 TM-score
    all_tm_scores = torch.stack(all_tm_scores, dim=1)  # [B, K]
    best_tm_scores = all_tm_scores.max(dim=1)[0]       # [B]

    # L2 损失计算
    pred_l2 = pred[:, :, 0, :]     # 仅对第一个候选坐标计算 L2 损失 [B, T, 3]
    target_l2 = target[:, :, 0, :] # [B, T, 3]
    l2_loss = F.mse_loss(pred_l2, target_l2)

    # 综合损失
    loss = -best_tm_scores.mean() + alpha * l2_loss
    return loss


def rna_mse_loss(pred, target):
    B, T, K, _ = pred.shape
    assert pred.shape == target.shape, "预测和目标的形状必须一致"
    pred_l2 = pred[:, :, 0, :]     # 仅对第一个候选坐标计算 L2 损失 [B, T, 3]
    target_l2 = target[:, :, 0, :] # [B, T, 3]
    l2_loss = F.mse_loss(pred_l2, target_l2)

    # 小 L2 正则化（可选）
    l2_reg = torch.mean(torch.norm(pred_l2, dim=2)) * 0.001

    # 综合损失
    loss = l2_loss + l2_reg
    return loss


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("⏹️ Early stopping triggered.")
                return True
            return False

def dynamic_alpha(epoch, switch_epoch=10, initial_alpha=10, final_alpha=1):
    """
    动态调整 alpha 参数，用于控制 L2 损失和 TM-score 损失的权重。
    
    参数:
    - epoch: 当前的训练 epoch。
    - switch_epoch: 切换到 TM-score 损失的 epoch。
    - initial_alpha: 初始阶段 L2 损失的权重。
    - final_alpha: 后期阶段 TM-score 损失的权重。
    
    返回:
    - 当前 epoch 的 alpha 值。
    """
    if epoch < switch_epoch:
        return initial_alpha  # 初始阶段以 L2 损失为主
    else:
        # 逐步减小 alpha，过渡到 TM-score 损失
        return max(final_alpha, initial_alpha - (epoch - switch_epoch) * (initial_alpha - final_alpha) / (100 - switch_epoch))

def RNA_3D_Predictor_Train(k_folds=5):
    train_data = load_rna_3d_data(TRAIN_SEQ_FILE_PATH, TRAIN_LABEL_FILE_PATH)
    # 验证集：全序列
    val_data = load_rna_3d_data(VALI_SEQ_FILE_PATH, VALI_LABEL_FILE_PATH)
    train_dataset = RNACoordsDataset(train_data)
    val_dataset = RNACoordsDataset(val_data)
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)
     # 初始化 KFold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"===== Fold {fold + 1}/{k_folds} =====")
         # 创建训练集和验证集
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=8, collate_fn=collate_fn)
    
        initmodel = SimpleCharTransformerWithRoPE(vocab_size).to(device)
        initmodel.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device,weights_only=True))
        initmodel.eval()
       
       
        trans_model = Trans3DPredictor().to(device)
        optimizer = optim.AdamW(trans_model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        early_stopping = EarlyStopping(patience=30)
        for epoch in range(1000):
            trans_model.train()
            total_loss = 0
            alpha = dynamic_alpha(epoch, switch_epoch=50, initial_alpha=10, final_alpha=1)

            for x, y, full_ids in tqdm(train_loader, desc=f"Fold {fold + 1}, Epoch {epoch+1} [Train]"):
                x, y,full_ids = x.to(device), y.to(device),full_ids.to(device)
                
                # 前向传播
                with torch.no_grad():
                    _ = initmodel(x)
                    char_emb = initmodel.embedding(x)  # [B, T, D]
                    last_hidden = initmodel.hidden_states[-1]  # [B, T, D]
                    mask = (x != 0).unsqueeze(-1).float()
                    valid_lengths = mask.sum(dim=1)  # [B, 1]
                    string_repr = (last_hidden * mask).sum(dim=1) / valid_lengths  # [B, D]
                    
                fused = char_emb + string_repr.unsqueeze(1).expand_as(char_emb)  # [B, T, D]              
                pred_coords,confidence = trans_model(fused)   
                loss = confidence_rmsd_loss(pred_coords,y,confidence)
                 # 调试信息
                # print(f"Loss: {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trans_model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            scheduler.step(avg_train_loss)
            # 打印当前学习率
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")
            print(f"  Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
            
            # 验证集评估
            trans_model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y, full_ids in tqdm(val_loader, desc=f"Fold {fold + 1},Epoch {epoch+1} [Val]"):
                    x, y, full_ids = x.to(device), y.to(device), full_ids.to(device)
                    with torch.no_grad():
                        _ = initmodel(x)
                        char_emb = initmodel.embedding(x)  # [B, T, D]
                        last_hidden = initmodel.hidden_states[-1]  # [B, T, D]
                        mask = (x != 0).unsqueeze(-1).float()
                        valid_lengths = mask.sum(dim=1)  # [B, 1]
                        string_repr = (last_hidden * mask).sum(dim=1) / valid_lengths  # [B, D]
                    fused = char_emb + string_repr.unsqueeze(1).expand_as(char_emb)  # [B, T, D]
                    pred_coords,confidence = trans_model(fused)     
                    # pred_coords = pred_coords * COORDS_STD + COORDS_MEAN
                    loss = weighted_tm_score_loss(pred_coords,y,alpha=alpha,confidence=confidence)
                    val_loss += loss.item()
                
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  Val Loss (pure TM-Score): {avg_val_loss:.4f}")
        
            if early_stopping.step(avg_val_loss):
                torch.save(trans_model.state_dict(), "rna_3d_model.pth")
                print("✅ New best model saved (rna_3d_model.pth)")
                break




def generate_submission_file(model, test_seq_file, output_path, k=5):
    test_df = pd.read_csv(test_seq_file)
    trans_model = Trans3DPredictor().to(device)
    trans_model.load_state_dict(torch.load(RNA_RNN_MODEL_FILE, map_location=device,weights_only=True))
    trans_model.eval()

    model.eval()
    results = []
    for _, row in test_df.iterrows():
        rna_id = row['target_id']
        seq = row['sequence']
        input_tensor = encode_string(seq, stoi).to(device)
        with torch.no_grad():
            _ = model(input_tensor)
            char_emb = model.embedding(input_tensor)
            last_hidden = model.hidden_states[-1]
            mask = (input_tensor != 0).unsqueeze(-1).float()
            valid_lengths = mask.sum(dim=1)
            string_repr = (last_hidden * mask).sum(dim=1) / valid_lengths
            fused = char_emb + string_repr.unsqueeze(1).expand_as(char_emb)  # [B, T, D]
            pred_coords,_ = trans_model(fused)
            pred_coords = pred_coords * COORDS_STD + COORDS_MEAN
        
        print(f"{rna_id} 预测坐标范围:", pred_coords.min().item(), pred_coords.max().item())
        pred_coords = pred_coords.squeeze(0).cpu().numpy()  # [T, K, 3]
        for i, base in enumerate(seq):
            coords = pred_coords[i]  # [K, 3]
            row_data = [f"{rna_id}_{i+1}", base, i+1]
            for xyz in coords:
                row_data.extend(xyz.tolist())
            results.append(row_data)

    columns = ['ID', 'resname', 'resid']
    for i in range(1, k + 1):
        columns += [f'x_{i}', f'y_{i}', f'z_{i}']
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"✅ Submission file saved to {output_path}")

def validate_rnn_model():
        # 加载验证集数据
        val_data = load_rna_3d_data(VALI_SEQ_FILE_PATH, VALI_LABEL_FILE_PATH)
        val_dataset = RNACoordsDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

        # 加载Transformer模型
        transformer_model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
        transformer_model.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device,weights_only=True))
        transformer_model.eval()

        # 加载RNN模型
        rnn_model = Trans3DPredictor().to(device)
        rnn_model.load_state_dict(torch.load(RNA_RNN_MODEL_FILE, map_location=device,weights_only=True))
        rnn_model.eval()

        # 计算验证集损失
        total_loss = 0
        with torch.no_grad():
            for x, y, full_ids in tqdm(val_loader, desc="Validating"):
                x, y, full_ids = x.to(device), y.to(device), full_ids.to(device)
                with torch.no_grad():
                    _ = transformer_model(x)
                    char_emb = transformer_model.embedding(x)  # [B, T, D]
                    last_hidden = transformer_model.hidden_states[-1]  # [B, T, D]
                    mask = (x != 0).unsqueeze(-1).float()
                    valid_lengths = mask.sum(dim=1)  # [B, 1]
                string_repr = (last_hidden * mask).sum(dim=1) / valid_lengths  # [B, D]
                fused = char_emb + string_repr.unsqueeze(1).expand_as(char_emb)  # [B, T, D]
               
               
                pred_coords,confidence = rnn_model(fused)
                loss = weighted_tm_score_loss(pred_coords, y,alpha=0,confidence=confidence)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

if __name__ == "__main__":
    RNA_3D_Predictor_Train()
   ## 训练完成后生成提交文件
    generate_submission_file(
        model=SimpleCharTransformerWithRoPE(vocab_size).to(device),
        test_seq_file=TEST_SEQ_FILE_PATH,
        output_path="submission.csv",
        k=K
    )
    
    # validate_rnn_model()
