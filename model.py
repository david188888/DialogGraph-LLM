import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# 使用本地模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./deepseek-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    output_attentions=True
)
model.generation_config = GenerationConfig.from_pretrained(model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "what is python?"
inputs = tokenizer(text, return_tensors="pt").to(device)
input_ids = inputs['input_ids']

# 获取模型输出，包括所有层的注意力输出
# outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

# 获取所有层的隐藏状态
# hidden_states = outputs.hidden_states
# print(f"\n总共有 {len(hidden_states)} 层隐藏状态(包括嵌入层)")
# for layer_idx, hidden_state in enumerate(hidden_states):
#     print(f"第 {layer_idx} 层隐藏状态形状: {hidden_state.shape}")

# # 获取所有层的注意力权重
# attentions = outputs.attentions
# print(f"\n总共有 {len(attentions)} 层注意力权重")
# for layer_idx, attention in enumerate(attentions):
#     print(f"第 {layer_idx} 层注意力权重形状: {attention.shape}")

# 获取最后一个token的预测概率分布
# logits = outputs.logits[:, -1, :]
# probs = F.softmax(logits, dim=-1)

# # 获取前5个最可能的token
# top_k = 5
# top_probs, top_indices = torch.topk(probs, top_k)


# 前向传播获取所有信息
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    hidden_states = [h.to(torch.float32).cpu() for h in outputs.hidden_states]  # 分层转换
    attentions = [a.to(torch.float32).cpu() for a in outputs.attentions]       # 分层转换
    logits = outputs.logits.to(torch.float32).cpu()
    # 添加注意力权重可视化

# 分析输出结构
print("输出对象包含的属性:", [k for k in outputs.keys()])

# 1. 输出张量（logits）分析
# logits = outputs.logits
print("\nLogits形状:", logits.shape)  # [batch_size, seq_len, vocab_size]

# 获取最后一个token的预测概率（典型的下一个token预测）
last_token_logits = logits[0, -1, :]
probs = torch.softmax(last_token_logits, dim=-1)

# 2. 每个token在模型中的变化可视化
# hidden_states = outputs.hidden_states  # 包含所有层的隐藏状态

# 选择要可视化的层（例如每3层）
selected_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
token_labels = tokenizer.convert_ids_to_tokens(input_ids[0])

plt.figure(figsize=(20, 15))
for i, layer_idx in enumerate(selected_layers):
    # 提取对应层的隐藏状态 [batch_size, seq_len, hidden_dim]
    layer_hidden = hidden_states[layer_idx][0].cpu().numpy()
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(layer_hidden)
    
    # 绘制散点图
    plt.subplot(4, 3, i+1)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=range(len(reduced)))
    plt.title(f"Layer {layer_idx} Representation")
    
    # 标注token文本
    for j, (x, y) in enumerate(reduced):
        plt.annotate(token_labels[j], (x, y), alpha=0.7)

plt.tight_layout()
plt.savefig('layer_representations.png')
plt.close()

# 3. 每个token的预测概率可视化（完整序列）
all_probs = torch.softmax(logits[0], dim=-1).cpu().numpy()

plt.figure(figsize=(15, 8))
for i, token in enumerate(token_labels):
    # 获取每个位置预测下个token的概率分布
    if i < all_probs.shape[0]-1:  # 最后一个位置没有对应的预测
        plt.plot(all_probs[i], alpha=0.7, label=f"Position {i} ({token})")
        # plt.savefig(f"token_probabilities_{i}.png")
        



plt.xlabel("Token ID")
plt.ylabel("Probability")
plt.title("Token Prediction Probabilities per Position")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('token_probabilities.png', bbox_inches='tight')

# 打印关键信息
print("\n关键信息汇总:")
print(f"输入文本: {text}")
print(f"Tokenized输入: {token_labels}")
print(f"最后一个token的top5预测:")
top5_probs, top5_ids = torch.topk(probs, 5)
for p, id in zip(top5_probs, top5_ids):
    print(f"- {tokenizer.decode(id)}: {p.item():.4f}")
#
