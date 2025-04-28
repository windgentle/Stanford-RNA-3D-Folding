"该项目是将字符串通过字符编码加transformer转换成向量形式"



import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import random
import string
from torchtune.modules import RotaryPositionalEmbeddings
from torch.utils.data import Dataset, DataLoader

#预
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


#准备数据

def generate_random_string_with_length_range(min_length, max_length, chars=string.ascii_letters + string.digits):
  """生成长度在指定范围内的随机字符串。

  Args:
    min_length: 随机字符串的最小长度（包含）。
    max_length: 随机字符串的最大长度（包含）。
    chars: 用于生成字符串的字符集，默认为大小写字母和数字。

  Returns:
    一个长度在指定范围内的随机字符串。
    如果 min_length 大于 max_length，则返回空字符串。
  """
  if min_length > max_length:
    return ""

  length = random.randint(min_length, max_length)
  random_string = ''.join(random.choice(chars) for _ in range(length))
  return random_string


#RNA中最基本的四种碱基为腺嘌呤（A）、尿嘧啶（U）、鸟嘌呤（G）、胞嘧啶（C）
#序列字符通道为4，不考虑其他信息，字符取ACGU
def generate_init_data(numbers=10000):
    # return [generate_random_string_with_length_range(4,10,chars=string.ascii_uppercase) for _ in range(numbers)]
    return [generate_random_string_with_length_range(4,256,chars="ACGUX-") for _ in range(numbers)]

#动态词表
# def build_vocab(data):
#     all_chars = sorted(set(''.join(data)))
#     stoi = {ch: i + 1 for i, ch in enumerate(all_chars)}  # +1 保留0做padding
#     stoi['<pad>'] = 0
#     itos = {i: ch for ch, i in stoi.items()}
#     return stoi, itos
# print(build_vocab("test"))

# vocab_size = len(stoi)  # 应该是 27


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

def collate_fn(batch):
    # 获取批次中所有序列的长度
    lengths = [len(x[0]) for x in batch]
    max_len = max(lengths)

    # 填充序列
    padded_x = [torch.cat([x[0], torch.zeros(max_len - len(x[0]), dtype=torch.long)]) for x in batch]
    padded_y = [torch.cat([x[1], torch.zeros(max_len - len(x[1]), dtype=torch.long)]) for x in batch]

    # 堆叠成张量
    x_tensor = torch.stack(padded_x)
    y_tensor = torch.stack(padded_y)
    return x_tensor, y_tensor
# 生成数据
# data = generate_init_data(10000)

# 构建词表
# stoi, itos = build_vocab(data)

# 固定 vocab（包含 <pad>）
VOCAB = ['<pad>'] + list("ACGUX-")
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

# 构建数据集与加载器
# dataset = CharDataset(data, stoi, max_len=12)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# for x, y in dataloader:
#     print("输入 shape:", x.shape)  # [batch_size, max_len]
#     print("标签 shape:", y.shape)  # [batch_size, max_len]
#     break

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
        self.rope = RotaryPositionalEmbeddings(dim=d_model // nhead)
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

def train():
    
    data = generate_init_data(10000)
    dataset = CharDataset(data, stoi)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,collate_fn=collate_fn)

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
        torch.save(model.state_dict(), "rna_transformer_model.pth")
        print("模型已保存为 rna_transformer_model.pth")

def encode_string(s, stoi):
    ids = [stoi[c] for c in s]
    return torch.tensor([ids], dtype=torch.long)  # shape [1, T]


def eval():
    model = SimpleCharTransformerWithRoPE(vocab_size).to(device)
    model.load_state_dict(torch.load("rna_transformer_model.pth", map_location=device))
    model.eval()  # 切换到推理模式（关闭 dropout 等）

    # 输入字符串
    input_str = "CGCGACCUCAGAUCAGACGUGGCGACCCGCUGAAUUUAAGCAUAUUAGUCAGCGGAGGAGAAGAAACUAACCAGGAUUCCCUCAGUAACGGCGAGUGAACAGGGAAGAGCCCAGCGCCGAAUCCCCGCCCCGCGGCGGGGCGCGGGACAUGUGGCGUACGGAAGACCCGCUCCCCGGCGCCGCUCGUGGGGGGCCCAAGUCCUUCUGAUCGAGGCCCAGCCCGUGGACGGUGUGAGGCCGGUAGCGGCCCCCGGCGCGCCGGGCCCGGGUCUUCCCGGAGUCGGGUUGCUUGGGAAUGCAGCCCAAAGCGGGUGGUAAACUCCAUCUAAGGCUAAAUACCGGCACGAGACCGAUAGUCAACAAGUACCGUAAGGGAAAGUUGAAAAGAACUUUGAAGAGAGAGUUCAAGAGGGCGUGAAACCGUUAAGAGGUAAACGGGUGGGGUCCGCGCAGUCCGCCCGGAGGAUUCAACCCGGCGGCGGGUCCGGCCGUGUCGGCGGCCCGGCGGAUCUUUCCCGCCCCCCGUUCCUCCCGACCCCUCCACCCGCCCUCCCUUCCCCCGCCGCCCCGGGCUCCGGCGGGUGCGGGGGUGGGCGGGCGGGGCCGGGGGUGGGGUCGGCGGGGGACCGUCCCCCGACCGGCGACCGGCCGCCGCCGGGCGCAUUUCCACCGCGGCGGUGCGCCGCGACCGGCUCCGGGACGGCUGGGAAGGCCCGGCGGGGAAGGUGGCUCGGGGGGCCCCGUCCGUCCGUCCGGCGGCGGCGGCGGCGGCGGGACCGAAACCCCCCCCGAGUGUUACAGCCCCCCCGGCAGCAGCACUCGCCGAAUCCCGGGGCCGAGGGAGCGAGACCCGUCGCCGCGCUCUCCCCCCUCCCGGCGCCCACCCCCGCGGGGACUCCCCCGCGGGGGCGCGCCGGCGUCUCCUCGUGGGGGGGCCGGGCCACCCCUCCCACGGCGCGACCGCUCUCCCACCCCUCCUCCCCGCGCCCCCGGGGUGCCGCGCGCGGGUCGGGGGGCGGGGCGGACUGUCCCCAGUGCGCCCCGGGCGGGUCGCGCCGUCGGGCCCGGGGGAGGUUCUCUCGGGGCCACGCGCGCGUCCCCCGAAGAGGGGGACGGCGGAGCGAGCGCACGGGGUCGGCGGCGACGUCGGCUACCCACCCGACCCGUCUUGAAACACGGACCAAGGAGUCUAACACGUGCGCGAGUCGGGGGCUCGCACGAAAGCCGCCGUGGCGCAAUGAAGGUGAAGGCCGGCGCGCUCGCCGGCCGAGGUGGGAUCCCGAGGCCUCUCCAGUCCGCCGAGGGCGCACCACCGGCCCGUCUCGCCCGCCGCGCCGGGGAGGUGGAGCACGAGCGCACGUGUUAGGACCCGAAAGAUGGUGAACUAUGCCUGGGCAGGGCGAAGCCAGAGGAAACUCUGGUGGAGGUCCGUAGCGGUCCUGACGUGCAAAUCGGUCGUCCGACCUGGGUAUAGGGGCGAAAGACUAAUCGAACCAUCUAGUAGCUGGUUCCCUCCGAAGUUUCCCUCAGGAUAGCUGGCGCUCUCGCAGACCCGACGCACCCCCGCCACGCAGUUUUAUCCGGUAAAGCGAAUGAUUAGAGGUCUUGGGGCCGAAACGAUCUCAACCUAUUCUCAAACUUUAAAUGGGUAAGAAGCCCGGCUCGCUGGCGUGGAGCCGGGCGUGGAAUGCGAGUGCCUAGUGGGCCACUUUUGGUAAGCAGAACUGGCGCUGCGGGAUGAACCGAACGCCGGGUUAAGGCGCCCGAUGCCGACGCUCAUCAGACCCCAGAAAAGGUGUUGGUUGAUAUAGACAGCAGGACGGUGGCCAUGGAAGUCGGAAUCCGCUAAGGAGUGUGUAACAACUCACCUGCCGAAUCAACUAGCCCUGAAAAUGGAUGGCGCUGGAGCGUCGGGCCCAUACCCGGCCGUCGCCGGCAGUCGAGAGUGGACGGGAGCGGCGGGGGCGGCGCGCGCGCGCGCGCGUGUGGUGUGCGUCGGAGGGCGGCCUCCUCCCGCCCACGCCCCGCUCCCCGCCCCCGGAGCCCCGCGGACGCUACGCCGCGACGAGUAGGAGGGCCGCUGCGGUGAGCCUUGAAGCCUAGGGCGCGGGCCCGGGUGGAG"
    input_tensor = encode_string(input_str, stoi).to(device)
    print(len(input_str))
    
    # 前向传播
    with torch.no_grad():
        logits = model(input_tensor)
    # 获取字符串整体编码（通过隐藏层平均）
    hidden = model.hidden_states[-1]  # [1, T, D]
    mask = (input_tensor != 0).unsqueeze(-1).float()
    valid_len = mask.sum(dim=1)
    string_repr = (hidden * mask).sum(dim=1) / valid_len  # [1, D]
    print("字符串整体编码向量:", string_repr)


if __name__ == "__main__":
    train()
    eval()
   