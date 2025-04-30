# 该 tm_score_loss 实现参考了 [Kaggle Ribonanza 比赛选手 Handsonlabs Software Academy] 的开源代码:https://www.kaggle.com/code/tobimichigan/rna-3d-structure-prediction-pipeline-algorithm

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


#const
device = 'mps' if torch.backends.mps.is_available() else 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

#数据预处理
# RNA_PARA_FILE = "/kaggle/input/rna-llm-parameter-file/"
RNA_PARA_FILE = ""
RNA_TRANS_MODEL_FILE = f"{RNA_PARA_FILE}rna_transformer_model.pth"
RNA_RNN_MODEL_FILE = f"{RNA_PARA_FILE}rna_3d_rnn_model.pth"
RNA_REPR_CACHE_FILE = f"{RNA_PARA_FILE}string_repr_cache.pt"
RNA_PERP_CACHE_VAL_FILE = f"{RNA_PARA_FILE}string_repr_cache_val.pt"

DATAPATH="stanford-rna-3d-folding"
TRAIN_SEQ_FILE_PATH = f"{DATAPATH}/train_sequences.v2.csv"
TRAIN_LABEL_FILE_PATH = f"{DATAPATH}/train_labels.v2.csv"
VALI_SEQ_FILE_PATH = f"{DATAPATH}/validation_sequences.csv"
VALI_LABEL_FILE_PATH = f"{DATAPATH}/validation_labels.csv"
TEST_SEQ_FILE_PATH = f"{DATAPATH}/test_sequences.csv"

# 标准化参数（提前设置好）
# x 坐标均值: 168.97385026312512, 方差: 123.40697723145699
# y 坐标均值: 171.8862117561932, 方差: 121.90471836458826
# z 坐标均值: 168.37776717440175, 方差: 126.73294381590155
COORDS_MEAN = torch.tensor([168.97, 171.89, 168.38], device=device)
COORDS_STD = torch.tensor([123.41, 121.90, 126.73], device=device)

"**************************************生成RNA序列原向量***********************************************"
#准备数据
def generate_random_string_with_length_range(min_length, max_length, chars=string.ascii_letters + string.digits):
  if min_length > max_length:
    return ""

  length = random.randint(min_length, max_length)
  random_string = ''.join(random.choice(chars) for _ in range(length))
  return random_string

#RNA中最基本的四种碱基为腺嘌呤（A）、尿嘧啶（U）、鸟嘌呤（G）、胞嘧啶（C）
#序列字符通道为6，不考虑其他信息，字符取ACGUX-
def generate_init_data(numbers=10000):
    # return [generate_random_string_with_length_range(4,10,chars=string.ascii_uppercase) for _ in range(numbers)]
    return [generate_random_string_with_length_range(4,256,chars="ACGUX-") for _ in range(numbers)]


class CharDataset(Dataset):
    def __init__(self, data, stoi):
        self.data = data
        self.stoi = stoi

    def __len__(self):
        return len(self.data)

    def encode(self, s):
        ids = [self.stoi[c] for c in s]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        s = self.data[idx]
        x = self.encode(s)
        y = x.clone() 
        return x, y


# 固定 vocab（包含 <pad>）
VOCAB = ['<pad>'] + list("ACGUX-")
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

#模型
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
    
# -------------------------------
# 5. 训练主流程
# -------------------------------

def rna_str_train():
    
    data = generate_init_data(10000)
    dataset = CharDataset(data, stoi)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # [B, T, V]
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # 打印隐藏层信息
        model.eval()
        sample_x, _ = next(iter(dataloader))
        sample_x = sample_x.to(device)
        with torch.no_grad():
            logits = model(sample_x)

        # print("=== 隐藏层输出 ===")
        # for layer_idx, h in enumerate(model.hidden_states):
        #     print(f"第 {layer_idx+1} 层: shape = {h.shape}")
        #     for i in range(2):  # 打印前两个样本的前几个token的向量
        #         valid_len = (sample_x[i] != 0).sum().item()
        #         print(f"样本 {i}，有效长度: {valid_len}")
        #         print(h[i, :valid_len].cpu().numpy())  # [T, D]
        # print("=" * 40)
        # # 每轮查看预测效果
        # model.eval()
        # sample_x, _ = next(iter(dataloader))
        # sample_x = sample_x.to(device)
        # with torch.no_grad():
        #     logits = model(sample_x)
        # preds = torch.argmax(logits, dim=-1)
        # for i in range(3):
        #     input_str = ''.join([itos[idx.item()] for idx in sample_x[i] if idx.item() != 0])
        #     output_str = ''.join([itos[idx.item()] for idx in preds[i] if idx.item() != 0])
        #     print(f"[{i}] input: {input_str}, output: {output_str}")
        # print("-" * 30)
        # 获取最后一层隐藏状态
        last_hidden = model.hidden_states[-1]  # [B, T, D]
        # 用 mask 去掉 padding
        mask = (sample_x != 0).unsqueeze(-1).float()  # [B, T, 1]
        masked_hidden = last_hidden * mask  # 仅保留非padding的向量

        # 每个样本有多少非padding token
        valid_lengths = mask.sum(dim=1)  # [B, 1]

        # 求平均编码（字符串整体表示）
        string_repr = masked_hidden.sum(dim=1) / valid_lengths  # [B, D]

        print("字符串整体编码 shape:", string_repr.shape)
        print("前两个字符串整体表示向量:")
        print(string_repr[:2].cpu().numpy())
            # 保存模型
        torch.save(model.state_dict(),RNA_TRANS_MODEL_FILE)
        print("模型已保存为 rna_transformer_model.pth")

def encode_string(s, stoi):
    ids = [stoi[c] for c in s]
    return torch.tensor([ids], dtype=torch.long)  # shape [1, T]


def rna_str_eval(input_str):
    model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
    model.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device))
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

"***************************************RNA处理************************************************"

def load_rna_3d_data(seq_file, label_file, max_len=256,stride=128):
    """
    从Stanford RNA数据集中加载RNA序列和对应3D坐标，支持mask处理NaN空值。
    返回(x, y, full_ids, mask)，用于后续loss屏蔽。
    """
    seq_df = pd.read_csv(seq_file)
    label_df = pd.read_csv(label_file)

    data = []
    for _, row in seq_df.iterrows():
        rna_id = row['target_id']
        sequence = row['sequence']
        seq_len = len(sequence)
        # 初始化全NaN的坐标数组
        coords = np.full((seq_len, 3), np.nan)
         # 提取 label 中属于当前 RNA 的坐标
        for _, coord_row in label_df[label_df['ID'].str.startswith(rna_id + "_")].iterrows():
            idx = int(coord_row['ID'].split("_")[-1])
            if 0<= idx < seq_len:
                coords[idx] = [coord_row['x_1'], coord_row['y_1'], coord_row['z_1']]
        
        # 生成有效坐标的 mask
        mask = ~np.isnan(coords).any(axis=-1)
        # 对有效坐标进行标准化
        coords[mask] = (coords[mask] - COORDS_MEAN.cpu().numpy()) / COORDS_STD.cpu().numpy()

        if seq_len != len(coords):
            continue  # 数据异常，跳过

        full_ids = torch.tensor([stoi.get(c, 0) for c in sequence], dtype=torch.long)

        # 处理整条序列（无滑窗）
        if max_len is None or stride is None:
            ids = [stoi.get(c, 0) for c in sequence]
            x = torch.tensor(ids, dtype=torch.long)
            K = 5  # 要和模型中的 K 对应
            coord_slice = np.repeat(coords[:, np.newaxis, :], K, axis=1)  # [T, 1, 3] -> [T, K, 3]
            y = torch.tensor(coord_slice, dtype=torch.float)
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            data.append((x, y, full_ids, mask_tensor))
            continue

        # 滑动窗口处理
        for start in range(0, seq_len, stride):
            
            end = start + max_len
            if start >= seq_len:
                break

            seq_slice = sequence[start:end]
            coord_slice = coords[start:end]
            mask_slice = mask[start:end]

            ids = [stoi.get(c, 0) for c in seq_slice]

            # padding 到 max_len
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids += [0] * pad_len
                coord_slice = np.vstack([coord_slice, np.full((pad_len, 3), np.nan)])
                mask_slice = np.concatenate([mask_slice, np.zeros(pad_len, dtype=bool)])
            else:
                ids = ids[:max_len]
                coord_slice = coord_slice[:max_len]
                mask_slice = mask_slice[:max_len]

            x = torch.tensor(ids, dtype=torch.long)
            coord_slice = np.repeat(coord_slice[:, np.newaxis, :], 5, axis=1)  # [T, 1, 3] -> [T, K, 3]
            y = torch.tensor(coord_slice, dtype=torch.float)
            mask_tensor = torch.tensor(mask_slice, dtype=torch.bool)
            data.append((x, y, full_ids, mask_tensor))

    return data

class RNACoordsDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for x, y, full_id, mask in data:
            if mask.sum() > 0:  # 只保留有效的样本
                self.data.append((x, y, full_id, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    xs, ys, full_ids,masks = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0.0)
    full_ids = pad_sequence(full_ids, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
    return xs, ys, full_ids,masks

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


#使用RNN做预测
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

def tm_score_loss(pred, target, mask):
    B, T, K, D = pred.shape

    mask = mask.unsqueeze(-1).expand(-1, -1, K)  # [B, T, K]

    squared_dists = torch.sum((pred - target) ** 2, dim=-1) + 1e-8  # [B, T, K]
    dists = torch.sqrt(torch.clamp(squared_dists, min=1e-8))        # [B, T, K]

    valid_lengths = mask[:, :, 0].sum(dim=1)  # [B]

    # 如果valid_lengths太小，强制调成至少1
    adjusted_len = torch.clamp(valid_lengths - 15, min=1e-6)

    d0 = 1.24 * adjusted_len ** (1/3) - 1.8  # [B]
    d0 = d0.clamp(min=0.5)                   # 防止负数！

    d0 = d0.view(B, 1, 1)                    # [B,1,1]

    tm_score_components = 1 / (1 + (dists / d0) ** 2)  # [B, T, K]
    tm_score_components = tm_score_components * mask  # mask无效点

    valid_counts = mask.sum(dim=1).clamp(min=1.0)  # [B,K]

    tm_scores = tm_score_components.sum(dim=1) / valid_counts  # [B,K]

    best_tm_scores = tm_scores.max(dim=1)[0]  # [B]

    # L2正则化
    pred_struct = pred[:, :, 0, :]
    pred_mask = mask[:, :, 0].unsqueeze(-1)
    pred_struct = pred_struct * pred_mask
    l2_reg = torch.mean(torch.norm(pred_struct, dim=2)) * 0.001

    return -best_tm_scores.mean() + l2_reg


def mixed_tm_l2_loss(pred, target, mask, alpha=1.0):

    # 调整形状: 统一为 [B, K, T, 3]
    pred = pred.permute(0, 2, 1, 3)     # [B, K, T, 3]
    target = target.permute(0, 2, 1, 3) # [B, K, T, 3]
    mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, pred.size(1), -1, 1)  # [B, K, T, 1]

    pred = pred * mask
    target = target * mask

    B, K, T, _ = pred.shape

    all_tm_scores = []
    for k in range(K):
        pred_struct = pred[:, k, :, :]     # [B, T, 3]
        target_struct = target[:, k, :, :] # [B, T, 3]

        # Pairwise L2 distance per point
        squared_dists = torch.sum((pred_struct - target_struct)**2, dim=-1) + 1e-8  # [B, T]
        dists = torch.sqrt(torch.clamp(squared_dists, min=1e-8))                   # [B, T]

        # TM-score 计算规则
        adjusted_len = max(T - 15, 1e-6)
        d0 = 1.24 * (adjusted_len ** (1/3)) - 1.8  # 常规 TM-score 长度调节因子

        tm_score_components = 1 / (1 + (dists / d0)**2)  # [B, T]

        tm_score = tm_score_components.mean(dim=1)  # [B]
        all_tm_scores.append(tm_score)

    # 所有K种预测中取最佳TM-score（更稳定）
    all_tm_scores = torch.stack(all_tm_scores, dim=1)  # [B, K]
    best_tm_scores = all_tm_scores.max(dim=1)[0]       # [B]

    # L2 Loss：我们只对 pred[:, 0] 和 target[:, 0] 做 MSE
    pred_l2 = pred[:, 0, :, :]     # [B, T, 3]
    target_l2 = target[:, 0, :, :] # [B, T, 3]
    l2_loss = F.mse_loss(pred_l2, target_l2)

    # 小 L2 正则（可选）
    l2_reg = torch.mean(torch.norm(pred_l2, dim=2)) * 0.001

    # 综合 loss
    return -best_tm_scores.mean() + l2_reg + alpha * l2_loss


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
        model.load_state_dict(torch.load(RNA_TRANS_MODEL_FILE, map_location=device))
        model.eval()
        # 预提取每条RNA序列的全局表示（训练集与验证集都要）
        string_repr_cache = build_string_repr_cache(model, train_subset)
        # 验证集也加入缓存
        string_repr_cache_val = build_string_repr_cache(model, val_subset, cache_path=RNA_PERP_CACHE_VAL_FILE)
        string_repr_cache.update(string_repr_cache_val)

        rnn_model = RNN3DPredictor(input_dim=512,k=5).to(device)
        conditioning = GlobalConditioning(dim=512, mode='gated').to(device)
        optimizer = optim.AdamW(rnn_model.parameters(), lr=1e-3)
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(100):
            rnn_model.train()
            total_loss = 0
                
            for x, y, full_ids,mask in tqdm(train_loader, desc=f"Fold {fold + 1}, Epoch {epoch+1} [Train]"):
                x, y,full_ids,mask = x.to(device), y.to(device),full_ids.to(device),mask.to(device)
            
                with torch.no_grad():
                    keys = [''.join([itos[idx.item()] for idx in ids if idx.item() != 0]) for ids in full_ids]
                    full_string_repr = torch.stack([string_repr_cache[k] for k in keys]).to(device)
                    char_emb = model.embedding(x)
                    fused = conditioning(char_emb, full_string_repr)
               
               
                pred_coords = rnn_model(fused)     
                loss = tm_score_loss(pred_coords, y,mask)
                 # 调试信息
                # print(f"Pred coords min: {pred_coords.min().item()}, max: {pred_coords.max().item()}")
                # print(f"Target coords min: {y.min().item()}, max: {y.max().item()}")
                # print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum().item()}")
                print(f"Loss: {loss.item()}")
            #     # 计算 loss（屏蔽掉 mask 中为 False 的位置）
            #     # 平方差
            #     loss = ((pred_coords - y) ** 2).sum(dim=-1)  # [B, T, K]
            #     # 应用 mask，扩展维度匹配 K
            #     loss = loss * mask.unsqueeze(-1)  # [B, T, K]
            #   # 总损失除以有效坐标数（防止NaN干扰）
            #     loss = loss.sum() / mask.sum()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")
            
            # 验证集评估
            rnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                # for x, y, full_ids in val_loader:
                for x, y, full_ids,mask in tqdm(val_loader, desc=f"Fold {fold + 1},Epoch {epoch+1} [Val]"):
                    x, y, full_ids,mask = x.to(device), y.to(device), full_ids.to(device),mask.to(device)
                    keys = [''.join([itos[idx.item()] for idx in ids if idx.item() != 0]) for ids in full_ids]
                    full_string_repr = torch.stack([string_repr_cache[k] for k in keys]).to(device)
                    char_emb = model.embedding(x)
                    fused = conditioning(char_emb, full_string_repr)
                    pred_coords = rnn_model(fused)
                    loss = tm_score_loss(pred_coords,y,mask)
                    val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                print(f"  Fold {fold + 1}, Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")
                if early_stopping.step(val_loss):
                    torch.save(rnn_model.state_dict(), RNA_RNN_MODEL_FILE)
                    print("✅ 提前停止，最佳 RNN 模型已保存为 rna_3d_rnn_model.pth")
                    break
            print(f"===== Fold {fold + 1} 完成 =====")
        print("🎉 K 折交叉验证完成！")


def build_string_repr_cache(model, dataset, cache_path=RNA_REPR_CACHE_FILE):
    if os.path.exists(cache_path):
        print(f"加载缓存的 string_repr 文件: {cache_path}")
        return torch.load(cache_path)

    print("生成 string_repr 缓存...")
    cache = {}
    model.eval()
    with torch.no_grad():
        for _, _, full_ids,_ in dataset:
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
    count_accum[count_accum == 0] = 1.0
    pred_coords_s = pred_coords_accum / count_accum

    # 反标准化
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



if __name__ == "__main__":
    RNA_3D_Predictor_Train()
    # 训练完成后生成提交文件
    generate_submission_file(
        model=SimpleCharTransformerWithRoPE(vocab_size).to(device),
        test_seq_file=TEST_SEQ_FILE_PATH,
        output_path="submission.csv",
        k=5
    )

