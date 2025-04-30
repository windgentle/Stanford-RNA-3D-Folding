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
RNA_RNN_MODEL_FILE = f"{RNA_PARA_FILE}rna_3d_rnn_model.pth"
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

    string_repr_expanded = string_repr.unsqueeze(1).expand_as(char_emb)  # [B, T, D]
    fused_char_repr = char_emb + string_repr_expanded  # [B, T, D]
    # print("字符串字符融合向量:", fused_char_repr)
    return fused_char_repr

#获取rna训练数据
def load_rna_3d_data(seq_file, label_file, max_len=256,stride=128):
    """
    从Stanford RNA数据集中加载RNA序列和对应3D坐标。
    """
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
        # 支持 max_len=None 或 stride=None 时直接处理整条序列
        if max_len is None or stride is None:
            ids = [stoi.get(c, 0) for c in sequence]
            x = torch.tensor(ids, dtype=torch.long)
            coord_slice = np.repeat(coords[:, np.newaxis, :], K, axis=1)  # [T, 1, 3] -> [T, K, 3]
            y = torch.tensor(coord_slice, dtype=torch.float)
            data.append((x, y, full_ids))
            continue
        for start in range(0, seq_len, stride):
            end = start + max_len
            if start >= seq_len:
                break
            seq_slice = sequence[start:end]
            coord_slice = coords[start:end]
            ids = [stoi.get(c, 0) for c in seq_slice]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids += [0] * pad_len
                coord_slice = np.vstack([coord_slice, np.zeros((pad_len, 3))])
            else:
                ids = ids[:max_len]
                coord_slice = coord_slice[:max_len]

            x = torch.tensor(ids, dtype=torch.long)
            coord_slice = np.repeat(coord_slice[:, np.newaxis, :], K, axis=1)  # [T, 1, 3] -> [T, K, 3]
            y = torch.tensor(coord_slice, dtype=torch.float)
            data.append((x, y,full_ids))
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

class GlobalConditioning(nn.Module):
    """
    将字符向量与全局向量通过非线性交互方式融合。
    支持三种方式：add（线性加和）、mlp（非线性MLP映射）、gated（门控融合）。
    """
    def __init__(self, dim, mode='mlp'):
        super().__init__()
        self.mode = mode
        if mode == 'mlp':
            self.fuse = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        elif mode == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
        # 'add' 模式不需要额外参数

    def forward(self, x, g):
        # x: [B, T, D], g: [B, D]
        B, T, D = x.shape
        g_expanded = g.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        if self.mode == 'add':
            return x + g_expanded
        elif self.mode == 'mlp':
            return self.fuse(torch.cat([x, g_expanded], dim=-1))
        elif self.mode == 'gated':
            gate = self.gate(torch.cat([x, g_expanded], dim=-1))
            return gate * x + (1 - gate) * g_expanded
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
#使用RNN模型做预测
class RNN3DPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256,k=5):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)
        self.output_layer = nn.Linear(hidden_dim * 2, 3 * k)
        self.k = k
    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # [B, T, 2H]

        rnn_out = self.dropout(rnn_out)

        out = self.output_layer(rnn_out)  # [B, T, 3*K]

        pred_coords = out.view(x.size(0), x.size(1), self.k, 3)  # [B, T, K, 3]
        return pred_coords

def tm_score_loss(pred, target, confidence=None):
    # target = target.permute(0, 2, 1, 3)
    pred_struct = pred[:, :, 0, :]
    all_tm_scores = []
    for i in range(target.shape[2]):
        target_struct = target[:, :, i, :]
        squared_dists = torch.sum((pred_struct - target_struct)**2, dim=-1) + 1e-8
        dists = torch.sqrt(torch.clamp(squared_dists, min=1e-8))
        seq_len = target.shape[1]
        adjusted_len = max(seq_len - 15, 1e-6)
        d0 = max(1.24 * adjusted_len**(1/3) - 1.8,0.5)

        tm_score_components = 1 / (1 + (dists / d0)**2)
      
        if confidence is not None:
            confidence_sum = confidence.sum(dim=1, keepdim=True)
            confidence_sum[confidence_sum == 0] = 1.0  # 避免除零
            norm_confidence = seq_len * confidence / confidence_sum
            tm_score_components = tm_score_components * norm_confidence
        tm_scores = tm_score_components.mean(dim=1)
        all_tm_scores.append(tm_scores)
    all_tm_scores = torch.stack(all_tm_scores, dim=1)
    best_tm_scores = all_tm_scores.max(dim=1)[0]
    l2_reg = torch.mean(torch.norm(pred_struct, dim=2)) * 0.001
    return -best_tm_scores.mean() + l2_reg

def mixed_tm_l2_loss(pred, target, alpha=10):
    """
    结合 TM-score 和 L2 损失的混合损失函数，适用于 RNA 3D 坐标预测。
    
    参数:
    - pred: 模型预测的坐标，形状为 [B, T, K, 3]
    - target: 真实的坐标，形状为 [B, T, K, 3]
    - alpha: L2 损失的权重系数
    
    返回:
    - 综合损失值
    """
    # 确保输入形状一致
    B, T, K, _ = pred.shape
    assert pred.shape == target.shape, "预测和目标的形状必须一致"

    # TM-score 计算
    all_tm_scores = []
    for k in range(K):
        pred_struct = pred[:, :, k, :]     # [B, T, 3]
        target_struct = target[:, :, k, :] # [B, T, 3]

        # Pairwise L2 distance per point
        squared_dists = torch.sum((pred_struct - target_struct)**2, dim=-1) + 1e-8  # [B, T]
        dists = torch.sqrt(torch.clamp(squared_dists, min=1e-8))                   # [B, T]

        # TM-score 计算规则
        seq_len = T  # 序列长度
        adjusted_len = max(seq_len - 15, 1e-6)  # 调整长度，避免负值
        d0 = max(1.24 * adjusted_len**(1/3) - 1.8, 0.5)  # TM-score 调节因子

        # 计算 TM-score 分量
        tm_score_components = 1 / (1 + (dists / d0)**2)  # [B, T]

        # 对序列取平均
        tm_scores = tm_score_components.mean(dim=1)  # [B]
        all_tm_scores.append(tm_scores)

    # 所有 K 种预测中取最佳 TM-score
    all_tm_scores = torch.stack(all_tm_scores, dim=1)  # [B, K]
    best_tm_scores = all_tm_scores.max(dim=1)[0]       # [B]

    # L2 损失计算
    pred_l2 = pred[:, :, 0, :]     # 仅对第一个候选坐标计算 L2 损失 [B, T, 3]
    target_l2 = target[:, :, 0, :] # [B, T, 3]
    l2_loss = F.mse_loss(pred_l2, target_l2)

    # 小 L2 正则化（可选）
    l2_reg = torch.mean(torch.norm(pred_l2, dim=2)) * 0.001

    # 综合损失
    loss = -best_tm_scores.mean() + alpha * l2_loss + l2_reg
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

def build_string_repr_cache(model, dataset, cache_path=RNA_REPR_CACHE_FILE):
    if os.path.exists(cache_path):
        print(f"加载缓存的 string_repr 文件: {cache_path}")
        return torch.load(cache_path)
    print("生成 string_repr 缓存...")
    cache = {}
    model.eval()
    with torch.no_grad():
        for _, _, full_ids in dataset:
            key = ''.join([itos[idx.item()] for idx in full_ids if idx.item() != 0])
            if key in cache:
                continue
            full_ids_tensor = full_ids.unsqueeze(0).to(device)
            _ = model(full_ids_tensor)
            hidden = model.hidden_states[-1]
            mask = (full_ids_tensor != 0).unsqueeze(-1).float()
            string_repr = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            cache[key] = string_repr.squeeze(0).detach().cpu()
    torch.save(cache, cache_path)
    print(f"缓存已保存到: {cache_path}")
    return cache

def RNA_3D_Predictor_Train(k_folds=5):
    # 训练集：滑动窗口切片
    train_data = load_rna_3d_data(TRAIN_SEQ_FILE_PATH, TRAIN_LABEL_FILE_PATH, max_len=256, stride=128)
    # 验证集：全序列
    val_data = load_rna_3d_data(VALI_SEQ_FILE_PATH, VALI_LABEL_FILE_PATH, max_len=None, stride=None)
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
    
        model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
        model.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device,weights_only=True))
        model.eval()
        # 预提取每条RNA序列的全局表示（训练集与验证集都要）
        train_cache_path = f"string_repr_cache_fold_{fold + 1}.pt"
        val_cache_path = f"string_repr_cache_val_fold_{fold + 1}.pt"
        string_repr_cache = build_string_repr_cache(model, train_subset,cache_path=train_cache_path)
        string_repr_cache_val = build_string_repr_cache(model, val_subset, cache_path=val_cache_path)
        string_repr_cache.update(string_repr_cache_val)

        rnn_model = RNN3DPredictor(input_dim=512,k=K).to(device)
        conditioning = GlobalConditioning(dim=512, mode='gated').to(device)
        optimizer = optim.AdamW(rnn_model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        early_stopping = EarlyStopping(patience=50)
        for epoch in range(1000):
            rnn_model.train()
            total_loss = 0
                
            for x, y, full_ids in tqdm(train_loader, desc=f"Fold {fold + 1}, Epoch {epoch+1} [Train]"):
                x, y,full_ids = x.to(device), y.to(device),full_ids.to(device)
                with torch.no_grad():
                    keys = [''.join([itos[idx.item()] for idx in ids if idx.item() != 0]) for ids in full_ids]
                    full_string_repr = torch.stack([string_repr_cache[k] for k in keys]).to(device)
                    char_emb = model.embedding(x)
                    fused = conditioning(char_emb, full_string_repr)
               
               
                pred_coords = rnn_model(fused)     
                loss = rna_mse_loss(pred_coords, y)
                 # 调试信息
                # print(f"Loss: {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            scheduler.step(avg_train_loss)
            # 打印当前学习率
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")
            print(f"  Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
            # print(f"  Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")
            
            # 验证集评估
            rnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y, full_ids in tqdm(val_loader, desc=f"Fold {fold + 1},Epoch {epoch+1} [Val]"):
                    x, y, full_ids = x.to(device), y.to(device), full_ids.to(device)
                    keys = [''.join([itos[idx.item()] for idx in ids if idx.item() != 0]) for ids in full_ids]
                    full_string_repr = torch.stack([string_repr_cache[k] for k in keys]).to(device)
                    # 滑动窗口推理
                    pred_coords,valid_mask = infer_with_sliding_window(
                        model=model,
                        x=x,
                        full_string_repr=full_string_repr,
                        conditioning=conditioning,
                        rnn_model=rnn_model,
                        window_size=256,
                        stride=128,
                        k=5
                    )
                    loss = tm_score_loss(pred_coords,y)
                    val_loss += loss.item()
                
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  Val Loss (pure TM-Score): {avg_val_loss:.4f}")
        
            if early_stopping.step(avg_val_loss):
                torch.save(rnn_model.state_dict(), "rna_3d_rnn_model.pth")
                print("✅ New best model saved (rna_3d_rnn_model.pth)")
                break


def infer_with_sliding_window(model, x, full_string_repr, conditioning, rnn_model,
                               window_size=256, stride=128, k=5):
    """
    使用滑动窗口推理并合并结果，返回反标准化坐标和有效 mask。
    """
    B, T = x.shape
    device = x.device

    pred_coords_accum = torch.zeros(B, T, k, 3, device=device)
    count_accum = torch.zeros(B, T, 1, device=device)

    if T <= window_size:
        # 序列长度比窗口小，直接整段处理
        with torch.no_grad():
            fused = conditioning(model.embedding(x), full_string_repr)
            pred_coords = rnn_model(fused)
            pred_coords = pred_coords * COORDS_STD + COORDS_MEAN
        valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        return pred_coords, valid_mask

    for start in range(0, T, stride):
        end = min(start + window_size, T)
        x_window = x[:, start:end]  # [B, cur_len]
        cur_len = end - start
        if cur_len == 0:
            continue

        with torch.no_grad():
            fused = conditioning(model.embedding(x_window), full_string_repr)
            pred_coords = rnn_model(fused)  # [B, cur_len, K, 3]

        # 累加到总坐标张量中
        pred_coords_accum[:, start:end] += pred_coords
        count_accum[:, start:end] += 1

    # 避免除0
    if (count_accum == 0).any():
        print("Warning: Some positions were not covered by the sliding window.")
    count_accum[count_accum == 0] = 1.0
    pred_coords_s = pred_coords_accum / count_accum.unsqueeze(-1).expand_as(pred_coords_accum)    # 反标准化
    pred_coords_s = pred_coords_s * COORDS_STD + COORDS_MEAN

    # 有效位置 mask
    valid_mask = (count_accum.squeeze(-1) > 0)

    return pred_coords_s, valid_mask

def rna_3D_eval(input_str,k=5):
    model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
    model.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device,weights_only=True))
    model.eval()  
    input_tensor = encode_string(input_str, stoi).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
    char_emb = model.embedding(input_tensor)
    last_hidden = model.hidden_states[-1]
    mask = (input_tensor != 0).unsqueeze(-1).float()
    valid_lengths = mask.sum(dim=1)  
    string_repr = (last_hidden * mask).sum(dim=1) / valid_lengths
    string_repr_expanded = string_repr.unsqueeze(1).expand_as(char_emb)
    fused_char_repr = char_emb + string_repr_expanded

    # 初始化 conditioning 和 rnn_model
    conditioning = GlobalConditioning(dim=512, mode='gated').to(device)
    rnn_model = RNN3DPredictor(input_dim=512, k=k).to(device)
    if os.path.exists("rna_3d_rnn_model.pth"):
        rnn_model.load_state_dict(torch.load(RNA_RNN_MODEL_FILE, map_location=device,weights_only=True))
    rnn_model.eval()

    pred_coords,valid_mask = infer_with_sliding_window(model, fused_char_repr, string_repr, conditioning, rnn_model, window_size=256, stride=128, k=k)
    return pred_coords,valid_mask

def generate_submission_file(model, test_seq_file, output_path, k=5):
    test_df = pd.read_csv(test_seq_file)
    rnn_model = RNN3DPredictor(input_dim=512, k=k).to(device)
    if os.path.exists("rna_3d_rnn_model.pth"):
        rnn_model.load_state_dict(torch.load(RNA_RNN_MODEL_FILE, map_location=device,weights_only=True))
    rnn_model.eval()
    conditioning = GlobalConditioning(dim=512, mode='gated').to(device)

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
            # fused = conditioning(char_emb, string_repr)
            pred_coords,valid_mask = infer_with_sliding_window(model, input_tensor, string_repr, conditioning, rnn_model, k=k)
        
        print(f"{rna_id} 预测坐标范围:", pred_coords.min().item(), pred_coords.max().item())
        pred_coords = pred_coords.squeeze(0).cpu().numpy()  # [T, K, 3]
        valid_mask = valid_mask.squeeze(0).cpu().numpy()    # [T]
        for i, base in enumerate(seq):
            if not valid_mask[i]:
                continue
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
        val_data = load_rna_3d_data(VALI_SEQ_FILE_PATH, VALI_LABEL_FILE_PATH, max_len=None, stride=None)
        val_dataset = RNACoordsDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

        # 加载Transformer模型
        transformer_model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
        transformer_model.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device,weights_only=True))
        transformer_model.eval()

        # 加载RNN模型
        rnn_model = RNN3DPredictor(input_dim=512, k=K).to(device)
        rnn_model.load_state_dict(torch.load(RNA_RNN_MODEL_FILE, map_location=device,weights_only=True))
        rnn_model.eval()

        # 初始化全局条件融合模块
        conditioning = GlobalConditioning(dim=512, mode='gated').to(device)

        # 计算验证集损失
        total_loss = 0
        with torch.no_grad():
            for x, y, full_ids in tqdm(val_loader, desc="Validating"):
                x, y, full_ids = x.to(device), y.to(device), full_ids.to(device)
                keys = [''.join([itos[idx.item()] for idx in ids if idx.item() != 0]) for ids in full_ids]
                full_string_repr = torch.stack(
                    [build_string_repr_cache(transformer_model, val_dataset)[k] for k in keys]
                ).to(device)
                char_emb = transformer_model.embedding(x)
                fused = conditioning(char_emb, full_string_repr)
                pred_coords = rnn_model(fused)
                loss = tm_score_loss(pred_coords, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

if __name__ == "__main__":
    RNA_3D_Predictor_Train()
    # 训练完成后生成提交文件
    generate_submission_file(
        model=SimpleCharTransformerWithRoPE(vocab_size).to(device),
        test_seq_file=TEST_SEQ_FILE_PATH,
        output_path="submission.csv",
        k=K
    )
    # validate_rnn_model()