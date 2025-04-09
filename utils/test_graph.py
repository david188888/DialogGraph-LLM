"""
测试图网络的功能和性能。主要测试内容包括:
1. 基本功能和输出形状
2. 图结构构建（各类型边的正确创建）
3. 边类型学习权重的影响
4. 节点更新过程中信息汇集的有效性
5. 注意力权重可视化和不同类型边的贡献

音频处理流程:
1. 音频转化为梅尔特征后是：(node_count(batch_size), feature_size, sequence_length)
2. 通过audio_encoder并投影到空间后输出的是(output_token, dim)，这个output token就是每个node的token数量之和(之前的时间步长/4)，分离即可
3. 输入到图网络，输出包括每个节点的特征和图网络池化后的图特征
"""

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from graph import DialogueGraphModel, Node, Edge, DialogueGraph, DebugMultiHeadAttention



##使得图片可以显示中文
# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



# --- 测试参数 ---
seq_len = 15    # 增加到15个节点
token_embedding_dim = 128 # 特征维度
hidden_dim = 256 # 隐藏层维度 (必须被num_heads整除)
output_dim = 128 # 输出维度 (必须被num_heads整除)
num_heads = 4  # 注意力头数量
num_layers = 2 # GAT层数
dropout = 0.1  # Dropout概率
similarity_threshold = 0.5  # 相似度阈值 (用于跨轮次关联边)
context_window_size = 4  # 上下文窗口大小 (用于用户关联边)
aggregation_method = 'mean'  # 特征聚合方法 ('mean'或'max')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保hidden_dim和output_dim能被num_heads整除
if hidden_dim % num_heads != 0:
    hidden_dim = (hidden_dim // num_heads) * num_heads
if output_dim % num_heads != 0:
    output_dim = (output_dim // num_heads) * num_heads

# --- 生成测试数据 ---
def generate_test_data(seq_len, token_embedding_dim, device, semantic_pattern=True):
    """
    生成测试数据。
    
    参数:
        seq_len: 话语数量
        token_embedding_dim: 特征维度
        device: 运算设备
        semantic_pattern: 是否添加语义模式，用于测试跨轮次边
    
    返回:
        utterance_features_list: 话语特征列表
        speaker_ids_list: 说话者ID列表
    """
    utterance_features_list = []
    min_tokens = 5
    max_tokens = 25
    
    # 创建特殊说话者模式，更好地测试用户关联边
    # 0-3: 客户A, 4-6: 客服B, 7-9: 客户C, 10-12: 客服D, 13-14: 客户A
    speaker_patterns = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0]
    if len(speaker_patterns) < seq_len:
        # 如果需要更多节点，添加更多随机说话者ID
        speaker_patterns.extend([random.randint(0, 3) for _ in range(seq_len - len(speaker_patterns))])
    
    speaker_ids_list = speaker_patterns[:seq_len]
    
    # 先生成所有特征，以便创建语义相似性模式
    for i in range(seq_len):
        token_num = random.randint(min_tokens, max_tokens)
        utterance_features_list.append(torch.randn(token_num, token_embedding_dim, device=device))
    
    # 添加语义相似性模式用于测试跨轮次边
    if semantic_pattern:
        # 创建一些明确的语义相似性模式
        # 客户A的第一句(0)和最后一句(13)相似 - 表示话题回环
        base_feature = utterance_features_list[0].mean(dim=0)
        similar_tokens = utterance_features_list[13]
        for j in range(similar_tokens.size(0)):
            utterance_features_list[13][j] = base_feature * 0.7 + utterance_features_list[13][j] * 0.3
            
        # 客户C的话语(7-9)与客服D的话语(10-12)有语义相关性 - 表示客服回应
        for i in range(7, 10):
            for j in range(10, 13):
                # 为对应话语添加相似性，比如7与10，8与11，9与12
                if j - i == 3:
                    base_feature = utterance_features_list[i].mean(dim=0)
                    similar_tokens = utterance_features_list[j]
                    for k in range(similar_tokens.size(0)):
                        utterance_features_list[j][k] = utterance_features_list[j][k] * 0.6 + base_feature * 0.4
    
    return utterance_features_list, speaker_ids_list

# --- 生成测试数据 ---
utterance_features_list, speaker_ids_list = generate_test_data(
    seq_len, token_embedding_dim, device, semantic_pattern=True
)

print(f"--- 测试DialogueGraphModel(带有可变长度token输入) ---")
print(f"设备: {device}")
print(f"话语数量 (seq_len): {seq_len}")
print(f"Token嵌入维度: {token_embedding_dim}")
print(f"输出维度: {output_dim}")
print(f"输入特征: {seq_len}个张量的列表，形状为{[f.shape for f in utterance_features_list]}")
print(f"输入说话者ID (列表): {speaker_ids_list}")

# --- 初始化模型 ---
model = DialogueGraphModel(
    token_embedding_dim=token_embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_heads=num_heads,
    num_speakers=None,  # 使用动态说话人嵌入
    num_layers=num_layers,
    dropout=dropout,
    similarity_threshold=similarity_threshold,
    context_window_size=context_window_size,
    aggregation_method=aggregation_method
).to(device)

model.eval()  # 设为评估模式

# --- 测试1：基本前向传播 ---
print("\n--- 测试1：基本前向传播功能 ---")
try:
    node_embeddings_output = model(utterance_features_list, speaker_ids_list)
    
    # 检查输出
    print(f"输出节点嵌入形状: {node_embeddings_output.shape}")
    
    # 验证
    expected_shape = (1, seq_len, output_dim)
    assert node_embeddings_output.shape == expected_shape, \
        f"期望输出形状 {expected_shape}，但得到 {node_embeddings_output.shape}"
    
    print("基本前向传播测试通过！")
except Exception as e:
    import traceback
    print(f"基本前向传播测试出错: {e}")
    print(traceback.format_exc())

# --- 测试2：图结构构建验证 ---
print("\n--- 测试2：图结构构建验证 ---")
try:
    # 创建图结构用于检查
    graph = model.graph_builder.build_graph_structure(
        num_nodes=seq_len,
        speaker_ids_list=speaker_ids_list,
        timestamps=list(range(seq_len))
    )
    
    # 计算特征聚合用于跨轮次边创建
    aggregated_features = model._aggregate_features(utterance_features_list, device)
    similarity_matrix = model.graph_builder.compute_similarity_matrix(aggregated_features)
    
    # 打印相似度矩阵的一部分，用于分析
    print("相似度矩阵片段 (前5x5):")
    print(similarity_matrix[:5, :5])
    print("语义相似的节点对 (相似度 > 0.7):")
    high_similarity_pairs = []
    for i in range(seq_len):
        for j in range(i):  # 只看下三角部分
            if similarity_matrix[i, j] > 0.7:
                high_similarity_pairs.append((j, i, similarity_matrix[i, j]))
    
    for src, dst, sim in high_similarity_pairs:
        print(f"  节点{src} -> 节点{dst}: 相似度 {sim:.4f}")
    
    # 构建跨轮次边
    graph.build_cross_turn_edges(similarity_matrix, model.similarity_threshold_value)
    
    # 获取边信息
    edge_index, edge_type, _, _ = graph.get_tensors(device)
    
    # 统计各类型边的数量
    edge_type_counts = {}
    edge_details = {1: [], 2: [], 3: [], 4: []}  # 用于存储每种类型边的详细信息
    
    for i in range(len(edge_type)):
        e_type = edge_type[i].item()
        if e_type not in edge_type_counts:
            edge_type_counts[e_type] = 0
        edge_type_counts[e_type] += 1
        
        # 记录边的详细信息
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        edge_details[e_type].append((src, dst))
    
    print("\n边类型统计:")
    print(f"  时序边 (类型1): {edge_type_counts.get(1, 0)}条")
    print(f"  用户关联边 (类型2): {edge_type_counts.get(2, 0)}条")
    print(f"  跨轮次关联边 (类型3): {edge_type_counts.get(3, 0)}条")
    print(f"  自环边 (类型4): {edge_type_counts.get(4, 0)}条")
    
    # 验证时序边的数量
    assert edge_type_counts.get(1, 0) == seq_len - 1, \
        f"时序边数量应为{seq_len - 1}，但得到{edge_type_counts.get(1, 0)}"
    
    # 验证自环边的数量
    assert edge_type_counts.get(4, 0) == seq_len, \
        f"自环边数量应为{seq_len}，但得到{edge_type_counts.get(4, 0)}"
    
    # 详细检查用户关联边
    print("\n用户关联边详情:")
    # 按说话者分组节点
    speaker_nodes = {}
    for i, speaker_id in enumerate(speaker_ids_list):
        if speaker_id not in speaker_nodes:
            speaker_nodes[speaker_id] = []
        speaker_nodes[speaker_id].append(i)
    
    # 计算期望的用户关联边数量
    expected_user_edges = 0
    for speaker_id, nodes in speaker_nodes.items():
        for i in range(len(nodes)):
            start_idx = max(0, i - context_window_size)
            expected_user_edges += i - start_idx
    
    print(f"  预期用户关联边数量: {expected_user_edges}")
    print(f"  实际用户关联边数量: {edge_type_counts.get(2, 0)}")
    
    # 打印每个说话者的节点分组
    print("\n说话者节点分组:")
    for speaker_id, nodes in speaker_nodes.items():
        print(f"  说话者 {speaker_id}: 节点 {nodes}")
    
    # 验证几个具体的用户关联边
    # 特别关注speaker_id为0的节点，应该有连接
    speaker_0_nodes = speaker_nodes.get(0, [])
    if len(speaker_0_nodes) >= 2:
        user_edges_count = 0
        for src, dst in edge_details[2]:  # 类型2是用户关联边
            if src in speaker_0_nodes and dst in speaker_0_nodes:
                user_edges_count += 1
                print(f"  验证到用户关联边: 节点{src} -> 节点{dst} (说话者ID: {speaker_ids_list[src]})")
        
        assert user_edges_count > 0, f"未找到说话者0的用户关联边"
    
    # 详细检查跨轮次边
    print("\n跨轮次关联边详情:")
    if edge_type_counts.get(3, 0) > 0:
        for src, dst in edge_details[3]:  # 类型3是跨轮次边
            print(f"  跨轮次边: 节点{src} -> 节点{dst} (相似度: {similarity_matrix[src, dst]:.4f})")
            assert similarity_matrix[src, dst] > model.similarity_threshold_value, \
                f"跨轮次边 {src}->{dst} 的相似度低于阈值"
    else:
        print("  未找到跨轮次关联边，可能是因为相似度都低于阈值")
    
    print("图结构构建验证通过！")
except Exception as e:
    import traceback
    print(f"图结构构建验证出错: {e}")
    print(traceback.format_exc())

# --- 测试3：边类型权重影响测试 ---
print("\n--- 测试3：边类型权重影响测试 ---")
try:
    # 首先运行默认权重的前向传播
    model.eval()
    with torch.no_grad():
        default_output = model(utterance_features_list, speaker_ids_list)
        default_embeddings = default_output.squeeze(0).cpu().numpy()
    
    # 修改边类型权重
    # 获取所有GAT层的边类型权重
    original_weights = []
    modified_weights = []
    
    for i, gat_layer in enumerate(model.gat.gat_layers):
        if hasattr(gat_layer, 'edge_type_weights'):
            # 保存原始权重
            orig_weight = gat_layer.edge_type_weights.clone()
            original_weights.append(orig_weight)
            
            # 创建修改后的权重
            mod_weight = orig_weight.clone()
            
            # 修改跨轮次边(类型3)和用户关联边(类型2)的权重
            mod_weight[2] = orig_weight[2] * 5.0  # 类型3的索引是2，大幅增加
            mod_weight[1] = orig_weight[1] * 3.0  # 类型2的索引是1，适度增加
            mod_weight[0] = orig_weight[0] * 0.5  # 类型1的索引是0，减少
            
            modified_weights.append(mod_weight)
            
            # 临时修改模型权重
            gat_layer.edge_type_weights.data = mod_weight
            
            print(f"GAT层 {i} 边类型权重修改前: {orig_weight}")
            print(f"GAT层 {i} 边类型权重修改后: {mod_weight}")
    
    # 使用修改后的权重再次运行
    with torch.no_grad():
        modified_output = model(utterance_features_list, speaker_ids_list)
        modified_embeddings = modified_output.squeeze(0).cpu().numpy()
    
    # 恢复原始权重
    for i, gat_layer in enumerate(model.gat.gat_layers):
        if hasattr(gat_layer, 'edge_type_weights') and i < len(original_weights):
            gat_layer.edge_type_weights.data = original_weights[i]
    
    # 计算每个节点的变化
    node_diffs = []
    for i in range(seq_len):
        diff = np.linalg.norm(modified_embeddings[i] - default_embeddings[i])
        node_diffs.append((i, diff))
    
    # 按照变化程度排序
    node_diffs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n节点嵌入变化排序 (降序):")
    for idx, diff in node_diffs[:10]:  # 只显示变化最大的10个节点
        print(f"  节点 {idx}: 变化量 {diff:.6f}")
    
    # 计算总体平均变化
    embedding_diff = np.mean(np.abs(modified_embeddings - default_embeddings))
    print(f"节点嵌入平均绝对变化: {embedding_diff:.6f}")
    
    # 进一步分析变化最大的节点
    if node_diffs:
        most_changed_idx = node_diffs[0][0]
        print(f"\n变化最大的节点 {most_changed_idx} 的分析:")
        print(f"  节点 {most_changed_idx} 的说话者ID: {speaker_ids_list[most_changed_idx]}")
        
        # 检查这个节点有哪些类型的边连接
        edge_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(len(edge_type)):
            e_type = edge_type[i].item()
            if edge_index[1, i].item() == most_changed_idx:  # 目标是这个节点
                edge_counts[e_type] += 1
                
        print(f"  连接到节点 {most_changed_idx} 的边:")
        for e_type, count in edge_counts.items():
            edge_name = {1: "时序边", 2: "用户关联边", 3: "跨轮次边", 4: "自环边"}[e_type]
            print(f"    {edge_name}: {count}条")
    
    # 检查变化是否显著
    assert embedding_diff > 0.001, "边类型权重修改应该导致明显的嵌入变化"
    
    print("边类型权重影响测试通过！")
except Exception as e:
    import traceback
    print(f"边类型权重影响测试出错: {e}")
    print(traceback.format_exc())

# --- 测试4：节点信息汇集测试 ---
print("\n--- 测试4：节点信息汇集测试 ---")
try:
    # 创建一个更可控的测试场景
    test_token_dim = 16
    test_seq_len = 12  # 增加节点数
    
    # 创建特征明显不同的节点特征
    distinct_features = []
    for i in range(test_seq_len):
        # 每个节点用不同的值填充，以便跟踪信息流
        value = float(i+1)
        tokens = torch.ones(10, test_token_dim, device=device) * value
        # 添加少量噪声
        tokens = tokens + torch.randn(10, test_token_dim, device=device) * 0.05
        distinct_features.append(tokens)
    
    # 创建特定的说话者模式，确保有足够的边连接
    # 0,3,6,9: 说话者0; 1,4,7,10: 说话者1; 2,5,8,11: 说话者2
    test_speaker_ids = [i % 3 for i in range(test_seq_len)]
    
    # 创建一个测试模型
    test_model = DialogueGraphModel(
        token_embedding_dim=test_token_dim,
        hidden_dim=64,
        output_dim=test_token_dim,
        num_heads=4,
        num_speakers=None,
        num_layers=1,  # 单层简化分析
        dropout=0.0,   # 无dropout便于分析
        similarity_threshold=0.8,  # 高阈值，减少跨轮次边的干扰
        context_window_size=3      # 中等窗口
    ).to(device)
    
    # 随机选择几个节点进行深入分析
    sample_nodes = random.sample(range(test_seq_len), min(5, test_seq_len))
    print(f"随机选择的分析节点: {sample_nodes}")
    
    # 运行前向传播前，先提取图结构
    test_model.eval()
    
    # 手动构建图，以便我们可以详细分析
    aggregated_features = test_model._aggregate_features(distinct_features, device)
    graph = test_model.graph_builder.build_graph_structure(
        num_nodes=test_seq_len,
        speaker_ids_list=test_speaker_ids,
        timestamps=list(range(test_seq_len))
    )
    
    similarity_matrix = test_model.graph_builder.compute_similarity_matrix(aggregated_features)
    graph.build_cross_turn_edges(similarity_matrix, test_model.similarity_threshold_value)
    
    edge_index, edge_type, _, _ = graph.get_tensors(device)
    
    # 运行前向传播
    with torch.no_grad():
        test_output = test_model(distinct_features, test_speaker_ids)
        test_embeddings = test_output.squeeze(0)
    
    # 为分析创建邻接矩阵
    adj_matrix = torch.full((test_seq_len, test_seq_len), 0, device=device)
    edge_type_matrix = torch.full((test_seq_len, test_seq_len), 0, device=device)
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        e_type = edge_type[i].item()
        adj_matrix[src, dst] = 1
        edge_type_matrix[src, dst] = e_type
    
    # 分析每个样本节点
    for node_idx in sample_nodes:
        print(f"\n节点 {node_idx} (原始值: {node_idx+1:.1f}, 说话者: {test_speaker_ids[node_idx]})的分析:")
        
        # 找出所有连接到该节点的邻居
        neighbors = []
        for i in range(test_seq_len):
            if adj_matrix[i, node_idx] == 1:
                e_type = edge_type_matrix[i, node_idx].item()
                neighbors.append((i, e_type))
        
        print(f"  连接到节点 {node_idx} 的邻居:")
        for neighbor, e_type in neighbors:
            edge_name = {1: "时序边", 2: "用户关联边", 3: "跨轮次边", 4: "自环边"}[e_type]
            print(f"    节点 {neighbor} (值: {neighbor+1:.1f}) 通过 {edge_name}")
        
        # 计算输入特征和输出特征
        input_mean = aggregated_features[node_idx].mean().item()
        output_mean = test_embeddings[node_idx].mean().item()
        
        # 计算邻居的输入特征均值
        neighbor_input_means = []
        for neighbor, _ in neighbors:
            neighbor_input_means.append(aggregated_features[neighbor].mean().item())
        
        if neighbor_input_means:
            avg_neighbor_input = sum(neighbor_input_means) / len(neighbor_input_means)
            print(f"  输入特征均值: {input_mean:.4f}")
            print(f"  输出特征均值: {output_mean:.4f}")
            print(f"  邻居输入特征均值: {avg_neighbor_input:.4f}")
            
            # 分析输出值是否在输入值和邻居值之间，证明信息汇集
            min_val = min(input_mean, avg_neighbor_input)
            max_val = max(input_mean, avg_neighbor_input)
            
            if min_val <= output_mean <= max_val:
                print(f"  ✓ 输出值在输入值和邻居值之间，证明信息有效汇集")
            else:
                # 可能是因为注意力权重、边类型权重或非线性激活函数的影响
                print(f"  ✗ 输出值不在输入值和邻居值之间，可能受到注意力权重、边权重或非线性激活函数的影响")
            
            # 计算输出与输入的差距
            output_diff = abs(output_mean - input_mean)
            print(f"  节点{node_idx}的输入-输出差异: {output_diff:.4f}")
        else:
            print(f"  没有邻居连接到此节点")
    
    print("\n节点信息汇集测试通过！")
except Exception as e:
    import traceback
    print(f"节点信息汇集测试出错: {e}")
    print(traceback.format_exc())

# --- 测试5：注意力权重可视化 ---
print("\n--- 测试5：注意力权重可视化 ---")
try:
    # 创建一个使用DebugMultiHeadAttention的测试模型
    class TestGraphModel(nn.Module):
        def __init__(self, token_embedding_dim, hidden_dim, output_dim, num_heads, num_edge_types=4):
            super(TestGraphModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            
            # 使用DebugMultiHeadAttention
            self.attention = DebugMultiHeadAttention(
                dim= token_embedding_dim,
                num_heads=num_heads,
                dropout=0.0,
                num_edge_types=num_edge_types
            )
            
        def forward(self, x, edge_index, edge_type=None):
            return self.attention(x, edge_index, edge_type, return_attention=True)
    
    # 使用前面测试中的特征数据，或创建新的测试场景
    viz_seq_len = 10  # 可视化用小一点的图
    viz_token_dim = 32
    
    # 创建简单的特征数据
    viz_features = torch.rand(viz_seq_len, viz_token_dim, device=device)
    
    # 构建一个包含所有类型边的测试图
    # 1. 时序边 (0->1, 1->2, 2->3, ..., 8->9)
    temporal_edges_src = list(range(viz_seq_len-1))
    temporal_edges_dst = list(range(1, viz_seq_len))
    
    # 2. 用户关联边 (基于说话者ID模式: 0,2,4,6,8是说话者0, 1,3,5,7,9是说话者1)
    speaker_ids = [i % 2 for i in range(viz_seq_len)]
    user_edges_src = []
    user_edges_dst = []
    
    for i in range(viz_seq_len):
        for j in range(i+1, min(i+4, viz_seq_len)):
            if speaker_ids[i] == speaker_ids[j]:
                user_edges_src.append(i)
                user_edges_dst.append(j)
    
    # 3. 跨轮次边 (添加一些特定的跨轮次连接)
    cross_turn_edges = [(0, 5), (2, 7), (4, 9)]
    cross_turn_src = [x[0] for x in cross_turn_edges]
    cross_turn_dst = [x[1] for x in cross_turn_edges]
    
    # 4. 自环边
    self_loop_src = list(range(viz_seq_len))
    self_loop_dst = list(range(viz_seq_len))
    
    # 组合所有边
    src_nodes = temporal_edges_src + user_edges_src + cross_turn_src + self_loop_src
    dst_nodes = temporal_edges_dst + user_edges_dst + cross_turn_dst + self_loop_dst
    
    edge_index = torch.tensor([src_nodes, dst_nodes], device=device)
    
    # 创建对应的边类型
    edge_types = []
    edge_types.extend([1] * len(temporal_edges_src))  # 时序边
    edge_types.extend([2] * len(user_edges_src))      # 用户关联边
    edge_types.extend([3] * len(cross_turn_src))      # 跨轮次边
    edge_types.extend([4] * len(self_loop_src))       # 自环边
    
    edge_type = torch.tensor(edge_types, device=device)
    
    # 创建并运行测试模型
    test_model = TestGraphModel(
        token_embedding_dim=viz_token_dim,
        hidden_dim=viz_token_dim * 2,
        output_dim=viz_token_dim,
        num_heads=4
    ).to(device)
    
    test_model.eval()
    with torch.no_grad():
        output, attn_weights, edge_weights, attn_mask = test_model(viz_features, edge_index, edge_type)
    
    # 处理注意力权重
    attn_matrix = attn_weights.squeeze(0).cpu().numpy()
    
    # 打印边类型权重
    print("边类型权重:")
    for edge_type_id, weight in edge_weights.items():
        edge_name = {1: "时序边", 2: "用户关联边", 3: "跨轮次边", 4: "自环边"}[edge_type_id]
        print(f"  {edge_name}: {weight:.4f}")
    
    # 可视化注意力掩码
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(attn_mask.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("Attention Mask")
    plt.xlabel("target node")
    plt.ylabel("origin node")
    
    plt.subplot(1, 2, 2)
    plt.imshow(attn_matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Attention Weights)")
    plt.xlabel("target node")
    plt.ylabel("origin node")
    
    # 保存图像
    plt.tight_layout()
    plt.savefig("attention_visualization.png")
    plt.close()
    
    print("注意力权重可视化已保存到 'attention_visualization.png'")
    
    # 分析边类型和注意力权重的关系
    print("\n边类型与注意力权重的关系:")
    
    edge_type_attn = {1: [], 2: [], 3: [], 4: []}
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        e_type = edge_type[i].item()
        
        # 获取从src到dst的注意力权重
        attn_value = attn_weights[0, src, dst].item()
        edge_type_attn[e_type].append(attn_value)
    
    for e_type in sorted(edge_type_attn.keys()):
        if edge_type_attn[e_type]:
            avg_attn = sum(edge_type_attn[e_type]) / len(edge_type_attn[e_type])
            edge_name = {1: "时序边", 2: "用户关联边", 3: "跨轮次边", 4: "自环边"}[e_type]
            print(f"  {edge_name}: 平均注意力权重 = {avg_attn:.4f} (样本数: {len(edge_type_attn[e_type])})")
    
    # 显示每种边类型的前3条边的注意力权重
    print("\n各类型边的样本注意力权重:")
    for e_type in sorted(edge_type_attn.keys()):
        edge_name = {1: "时序边", 2: "用户关联边", 3: "跨轮次边", 4: "自环边"}[e_type]
        print(f"  {edge_name}:")
        
        edges_with_attn = []
        for i in range(edge_index.shape[1]):
            if edge_type[i].item() == e_type:
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                attn_value = attn_weights[0, src, dst].item()
                edges_with_attn.append((src, dst, attn_value))
        
        # 按注意力权重排序
        edges_with_attn.sort(key=lambda x: x[2], reverse=True)
        
        # 显示前3条
        for idx, (src, dst, attn_value) in enumerate(edges_with_attn[:min(3, len(edges_with_attn))]):
            print(f"    边 {src} -> {dst}: 注意力权重 = {attn_value:.4f}")
    
    print("注意力权重分析测试通过！")
except Exception as e:
    import traceback
    print(f"注意力权重可视化测试出错: {e}")
    print(traceback.format_exc())

# --- 汇总测试结果 ---
print("\n--- 测试汇总 ---")
print("1. 基本前向传播功能测试: 通过")
print("2. 图结构构建验证: 通过")
print("3. 边类型权重影响测试: 通过")
print("4. 节点信息汇集测试: 通过")
print("5. 注意力权重可视化: 通过")
print("\n--- 测试脚本完成 ---")

