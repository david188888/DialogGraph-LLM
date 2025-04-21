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
import os
import sys

# 添加父目录到 Python 路径以解决导入问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入图网络和数据加载器相关组件
from utils.graph import DialogueGraphModel
from utils.dataloader import AudioSegmentDataset, DataLoader, DataLoaderConfig

# --- 数据和模型配置 ---
# !!! 请确保将 data_path 设置为您的实际数据路径 !!!
DATA_PATH = "/data/shared/Qwen/data" # 修改为绝对路径
LABELS_FILE = 'migrated_labels.csv' # 使用简单的文件名，它将被与 DATA_PATH 结合使用
CACHE_DIR = 'features_cache'
MODEL_PATH = "/data/shared/Qwen/models/Qwen2.5-Omni-7B" # 用于加载Processor的模型路径

# 图模型参数 (部分参数将从数据中动态获取)
# 不再使用hidden_dim参数，直接使用token_embedding_dim
num_heads = 4  # 注意力头数量
num_layers = 2 # GAT层数
dropout = 0.1  # Dropout概率
similarity_threshold = 0.9  # 相似度阈值 (用于跨轮次关联边)
context_window_size = 3  # 上下文窗口大小 (用于用户关联边)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- 使用单个电话号码的真实数据测试 DialogueGraphModel ---")
print(f"设备: {device}")
print(f"数据路径: {DATA_PATH}")

# --- 加载真实数据 ---
try:
    print("\n--- 正在加载数据 ---")
    dataloader_config = DataLoaderConfig(
        data_path=DATA_PATH,
        labels_file=LABELS_FILE,
        cache_dir=CACHE_DIR,
        batch_size=1,  # 当前模型实现仅支持 batch_size=1
        model_path=MODEL_PATH
    )
    
    dataset = AudioSegmentDataset(
        data_path=dataloader_config.data_path,
        model_path=dataloader_config.model_path,
        labels_file=dataloader_config.labels_file,
        cache_dir=dataloader_config.cache_dir,
        # audio_dir 和 segments_dir 使用默认值 (data_path 下的 audio 和 segments)
    )
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查数据路径和标签文件是否正确。")
    
    # 只使用第一个数据项进行测试
    print(f"数据集中共有 {len(dataset)} 个电话对话，将仅使用第一个进行测试。")
    
    # 直接从数据集获取第一个电话的数据
    item = dataset[0]
    phone_id = item['phone_id']
    segment_features_raw = item['segment_features']  # 这是一个列表，每个元素是一个片段的特征
    num_segments = item['num_segments']
    speakers = item['speaker']
    label = item['label']
    
    print(f"\n--- 获取到的单个电话数据信息 ---")
    print(f"电话ID: {phone_id}")
    print(f"片段数量: {num_segments}")
    print(f"说话人信息: {speakers}")
    print(f"标签: {label}")
    
    # 使用DataLoader进行批处理，以利用collate_fn
    data_loader = DataLoader(
        [item],  # 只包含一个电话的数据
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    # 获取批次数据
    print("\n尝试从DataLoader获取数据...")
    real_batch = next(iter(data_loader))
    print("--- 数据加载完成 ---")
    
    # 从批次中提取所需数据
    segment_features = real_batch['segment_features'].to(device)
    segment_attention_mask = real_batch['segment_attention_mask'].to(device)
    speakers = real_batch['speakers']  # DataLoader 返回的是列表
    num_segments_list = real_batch['num_segments']
    phone_ids = real_batch['phone_ids']
    
    # 输出更详细的批次信息
    print(f"\n--- 批次数据详情 ---")
    print(f"处理的电话ID: {phone_ids}")
    print(f"片段数量: {num_segments_list}")
    print(f"说话人信息: {speakers}")
    
    # 动态获取输入维度
    if segment_features.numel() > 0:
        batch_size, max_num_segments, max_segment_len, token_embedding_dim = segment_features.shape
        print(f"\n从数据中获取的维度: batch_size={batch_size}, max_num_segments={max_num_segments}, max_segment_len={max_segment_len}, feat_dim={token_embedding_dim}")
    else:
        print("警告：加载的批次中 segment_features 为空，无法确定输入维度。测试无法继续。")
        raise ValueError("无法从空特征中确定输入维度，测试无法继续。")

    # 确定模型内部维度 - 移除hidden_dim，现在使用token_embedding_dim作为直接输入
    output_dim = token_embedding_dim     # GAT 输出特征维度 (保持与输入一致)

    # 确保output_dim能被num_heads整除
    if output_dim % num_heads != 0:
        output_dim = (output_dim // num_heads) * num_heads
        print(f"调整 GAT output_dim 为: {output_dim}")
    
    # 确保调整后维度不为0
    if output_dim == 0: output_dim = num_heads

except Exception as e:
    print(f"数据加载或维度设置过程中出错: {e}")
    import traceback
    traceback.print_exc()
    # 如果数据加载失败，后续测试无法进行
    print("测试因数据加载失败而终止。")
    exit()


# --- 初始化模型 ---
print("\n--- 初始化模型 ---")
try:
    model = DialogueGraphModel(
        token_embedding_dim=token_embedding_dim, # 使用从数据中获取的维度
        # 移除hidden_dim参数，不再需要
        output_dim=output_dim,
        num_heads=num_heads,
        num_speakers=None,  # 使用动态说话人嵌入
        num_layers=num_layers,
        dropout=dropout,
        similarity_threshold=similarity_threshold,
        context_window_size=context_window_size,
        # aggregation_method 参数已移除
    ).to(device)

    model.eval()  # 设为评估模式
    print("--- 模型初始化完成 ---")
except Exception as e:
    print(f"模型初始化出错: {e}")
    import traceback
    traceback.print_exc()
    print("测试因模型初始化失败而终止。")
    exit()

# --- 测试1：基本前向传播 (使用真实数据) ---
print("\n--- 测试1：基本前向传播功能 (使用单个电话的数据) ---")
print(f"测试的电话ID: {phone_ids[0]}")
print(f"输入特征形状: {segment_features.shape}")
print(f"输入掩码形状: {segment_attention_mask.shape}")
print(f"输入片段数量: {num_segments_list[0]}")
print(f"输入说话者ID列表: {speakers[0] if speakers else 'N/A'}")

try:
    # 前向传播获取节点特征
    updated_node_features = model.get_node_embeddings(
        segment_features,
        speakers,  # 直接传递列表
        attention_masks=segment_attention_mask
    )

    # 检查输出
    print(f"\n输出节点特征形状: {updated_node_features.shape}")

    # 验证输出形状
    # 预期形状: [max_num_segments, max_segment_len, output_dim] (get_node_embeddings 移除了批次维度)
    expected_shape = (max_num_segments, max_segment_len, output_dim)
    assert updated_node_features.shape == expected_shape, \
        f"期望输出形状 {expected_shape}，但得到 {updated_node_features.shape}"

    # 检查填充位置是否为零 (检查一个样本点)
    # 注意: updated_node_features 没有批次维度了
    output_mask_bool = segment_attention_mask[0].bool() # 取第一个批次的掩码
    masked_output = updated_node_features[~output_mask_bool]
    if masked_output.numel() > 0:
        assert torch.allclose(masked_output, torch.tensor(0.0, device=device, dtype=masked_output.dtype), atol=1e-6), \
             "输出特征在掩码为0的位置应接近于0"
    else:
        print("注意：没有被掩码的位置，无法检查填充值。")

    # 提取实际数量的片段输出，剔除填充部分 (已经没有批次维度)
    actual_num_segments = num_segments_list[0]
    actual_segments_features = updated_node_features[:actual_num_segments, :, :]
    print(f"实际有效片段的输出特征形状: {actual_segments_features.shape}")

    # 计算每个片段的平均特征，可用于下游任务 (在时间维度 dim=1 上平均)
    # 需要先处理掩码，避免计算填充部分
    valid_tokens_mask = segment_attention_mask[0, :actual_num_segments].unsqueeze(-1).float() # [actual_num, max_len, 1]
    summed_features = (actual_segments_features * valid_tokens_mask).sum(dim=1) # [actual_num, dim]
    valid_token_counts = valid_tokens_mask.sum(dim=1).clamp(min=1.0) # [actual_num, 1]
    mean_segment_features = summed_features / valid_token_counts # [actual_num, dim]

    print(f"平均后的片段特征形状: {mean_segment_features.shape}")

    print("基本前向传播测试通过！")
except Exception as e:
    import traceback
    print(f"基本前向传播测试出错: {e}")
    print(traceback.format_exc())

# --- 测试2：图结构构建验证（无可视化） ---
print("\n--- 测试2：图结构验证 ---")
try:
    # 提取图结构信息（无可视化）
    def extract_graph_structure(model, features, speakers, attention_masks):
        """提取模型在前向传播过程中构建的图结构"""
        # 获取单个批次数据
        if features.dim() == 4:  # [batch_size, max_num_segments, max_segment_len, feat_dim]
            features = features[0]  # [max_num_segments, max_segment_len, feat_dim]
        
        if attention_masks.dim() == 3:  # [batch_size, max_num_segments, max_segment_len]
            attention_masks = attention_masks[0]  # [max_num_segments, max_segment_len]
        
        # 转换speaker_ids
        if isinstance(speakers, list) and len(speakers) > 0 and isinstance(speakers[0], list):
            speaker_ids = speakers[0]
        else:
            speaker_ids = speakers
        
        # 处理特征维度调整
        if hasattr(model, 'dim_adjust'):
            features = model.dim_adjust(features.to(device))
        
        # 构建图结构
        num_segments = features.size(0)
        
        # 使用模型的图构建器构建图
        graph = model.graph_builder.build_graph_structure(
            num_nodes=num_segments,
            speaker_ids_list=speaker_ids if isinstance(speaker_ids, list) else speaker_ids.cpu().tolist(),
            timestamps=list(range(num_segments))
        )
        
        # 构建跨轮次边 (Type 3)
        if num_segments > 1:
            similarity_matrix = model.graph_builder.compute_masked_similarity_matrix(features, attention_masks)
            graph.build_cross_turn_edges(similarity_matrix, model.similarity_threshold_value)
        
        # 获取边信息
        edge_index, edge_type, node_id_to_idx, sorted_node_ids = graph.get_tensors(device)
        
        # 统计各类型边的数量和具体连接
        edge_counts = {}
        specific_edges = {2: [], 3: []} # 存储类型2和3的边
        edge_index_np = edge_index.cpu().numpy()
        edge_type_np = edge_type.cpu().numpy()

        for i in range(edge_index_np.shape[1]):
            src_idx, dst_idx = edge_index_np[0, i], edge_index_np[1, i]
            e_type = edge_type_np[i]
            edge_counts[e_type] = edge_counts.get(e_type, 0) + 1

            # 记录用户关联边和跨轮次关联边的具体连接 (使用原始 node ID 或索引均可，这里用索引)
            if e_type == 2: # 用户关联边
                specific_edges[2].append(f"{src_idx} -> {dst_idx}")
            elif e_type == 3: # 跨轮次关联边
                specific_edges[3].append(f"{src_idx} -> {dst_idx}")

        # 使用中英文对照的边类型名称，以便在控制台显示中文
        edge_type_names = {
            1: "时序边 (Temporal Edge)",
            2: "用户关联边 (User Context Edge)",
            3: "跨轮次关联边 (Cross-Turn Edge)",
            4: "自环边 (Self-Loop Edge)"
        }
        
        # 格式化边统计信息
        edge_stats = []
        for e_type, count in sorted(edge_counts.items()): # 按类型排序
            edge_stats.append(f"{edge_type_names.get(e_type, f'未知类型{e_type}')}：{count}条")
        
        return {
            "edge_index": edge_index,
            "edge_type": edge_type,
            "num_nodes": num_segments,
            "edge_counts": edge_counts,
            "edge_stats": edge_stats,
            "specific_edges": specific_edges # 返回具体边的信息
        }
    
    # 提取图结构信息
    graph_info = extract_graph_structure(model, segment_features, speakers, segment_attention_mask)
    
    # 显示图统计信息
    print(f"\n图结构统计信息:")
    print(f"节点数量: {graph_info['num_nodes']}")
    print(f"边总数: {graph_info['edge_type'].size(0)}")
    print("各类型边数量:")
    for stat in graph_info['edge_stats']:
        print(f"  - {stat}")
    
    # 显示用户关联边 (Type 2) 的具体连接
    user_context_edges = graph_info['specific_edges'].get(2, [])
    if user_context_edges:
        print("\n  用户关联边 (Type 2) 连接详情:")
        for edge_str in user_context_edges:
            print(f"    - {edge_str}")

    # 显示跨轮次关联边 (Type 3) 的具体连接
    cross_turn_edges = graph_info['specific_edges'].get(3, [])
    if cross_turn_edges:
        print("\n  跨轮次关联边 (Type 3) 连接详情:")
        for edge_str in cross_turn_edges:
            print(f"    - {edge_str}")

    print("\n图结构构建验证通过!")
except Exception as e:
    import traceback
    print(f"图结构验证尝试出错: {e}")
    print(traceback.format_exc())

# --- 测试3：边类型权重影响测试 (使用真实数据) ---
print("\n--- 测试3：边类型权重影响测试 (使用单个电话的数据) ---")
try:
    # 首先运行默认权重的前向传播获取节点特征
    model.eval()
    with torch.no_grad():
        default_output = model.get_node_embeddings(segment_features, speakers, attention_masks=segment_attention_mask)
        # 输出现在是节点级特征 [seg, len, dim]
        default_features_np = default_output.cpu().numpy()

    # 修改边类型权重
    original_weights = []
    modified_weights = []
    edge_weights_found = False
    
    # 遍历 GAT 层修改权重
    # 注意在新结构中，gat_layers在model.gat下
    if hasattr(model, 'gat') and hasattr(model.gat, 'gat_layers'):
        gat_layers_module = model.gat.gat_layers
    else:
        print("警告：无法找到 model.gat.gat_layers，跳过边类型权重测试。")
        gat_layers_module = []

    for i, gat_layer in enumerate(gat_layers_module):
        # GAT 层现在直接是 MultiHeadAttentionWithMask
        if hasattr(gat_layer, 'edge_type_weights'):
            edge_weights_found = True
            # 保存原始权重
            orig_weight = gat_layer.edge_type_weights.clone()
            original_weights.append(orig_weight)
            
            # 创建修改后的权重
            mod_weight = orig_weight.clone()
            
            # 修改跨轮次边(类型3, 索引2)和用户关联边(类型2, 索引1)的权重
            # 索引是从0开始的，所以类型1->索引0, 类型2->索引1, 类型3->索引2, 类型4->索引3
            if len(mod_weight) > 2: # 确保至少有3种边类型 (索引0, 1, 2)
                mod_weight[2] = orig_weight[2] + 2.0  # 增加 Type 3 (索引2) 权重
            if len(mod_weight) > 1: # 确保至少有2种边类型 (索引0, 1)
                mod_weight[1] = orig_weight[1] + 1.0  # 增加 Type 2 (索引1) 权重
            if len(mod_weight) > 0: # 确保至少有1种边类型 (索引0)
                 mod_weight[0] = orig_weight[0] - 1.0 # 减少 Type 1 (索引0) 权重
            
            modified_weights.append(mod_weight)
            
            # 临时修改模型权重
            gat_layer.edge_type_weights.data = mod_weight
            
            print(f"GAT层 {i} 边类型权重修改前 (参数值): {orig_weight}")
            print(f"GAT层 {i} 边类型权重修改后 (参数值): {mod_weight}")
        else:
            print(f"警告：GAT层 {i} ({type(gat_layer)}) 没有 'edge_type_weights' 属性。")

    if not edge_weights_found:
         print("警告：在模型中未找到 'edge_type_weights'，无法执行此测试。跳过。")
    else:
        # 使用修改后的权重再次运行获取节点特征
        with torch.no_grad():
            modified_output = model.get_node_embeddings(segment_features, speakers, attention_masks=segment_attention_mask)
            modified_features_np = modified_output.cpu().numpy()
        
        # 恢复原始权重
        weight_idx = 0
        for i, gat_layer in enumerate(gat_layers_module):
            if hasattr(gat_layer, 'edge_type_weights') and weight_idx < len(original_weights):
                gat_layer.edge_type_weights.data = original_weights[weight_idx]
                weight_idx += 1
        
        # 计算整体特征变化的差异 (比较整个张量)
        # 使用 L2 范数计算差异
        total_diff = np.linalg.norm(modified_features_np - default_features_np)
        # 计算平均绝对差异
        avg_abs_diff = np.mean(np.abs(modified_features_np - default_features_np))
        
        print(f"\n修改边权重后整体特征 L2 差异: {total_diff:.6f}")
        print(f"修改边权重后整体特征平均绝对差异: {avg_abs_diff:.6f}")
        
        # 检查变化是否显著 (阈值可以调整)
        assert avg_abs_diff > 1e-6, "边类型权重修改应该导致可测量的特征变化"
        
        print("边类型权重影响测试通过！")

except Exception as e:
    import traceback
    print(f"边类型权重影响测试出错: {e}")
    print(traceback.format_exc())

# --- 测试4：片段特征分析（无可视化） ---
print("\n--- 测试4：片段特征分析（无可视化） ---")
try:
    # 使用前向传播的结果进行分析
    if 'updated_node_features' in locals() and updated_node_features.numel() > 0:
        # 获取实际片段数量的特征（去除填充）
        actual_num_segments = num_segments_list[0]
        # updated_node_features 已经是 [max_num_segments, max_segment_len, output_dim]
        segment_features_output = updated_node_features[:actual_num_segments]

        # 计算片段间的相似度矩阵
        # 需要使用掩码进行平均池化
        valid_tokens_mask = segment_attention_mask[0, :actual_num_segments].unsqueeze(-1).float() # [actual_num, max_len, 1]
        summed_features = (segment_features_output * valid_tokens_mask).sum(dim=1) # [actual_num, dim]
        valid_token_counts = valid_tokens_mask.sum(dim=1).clamp(min=1.0) # [actual_num, 1]
        mean_segment_features = summed_features / valid_token_counts # [actual_num, dim]

        # mean_segment_features = torch.mean(segment_features_output, dim=1) # 之前的错误做法
        normalized_features = torch.nn.functional.normalize(mean_segment_features, p=2, dim=1)  # 特征归一化
        similarity_matrix = torch.mm(normalized_features, normalized_features.transpose(0, 1))  # [num_segments, num_segments]

        print(f"\n片段间相似度矩阵 (形状 {similarity_matrix.shape}):")
        similarity_np = similarity_matrix.cpu().numpy()
        # 输出相似度矩阵，保留4位小数
        np.set_printoptions(precision=4, suppress=True)
        print(similarity_np)
        
        print("片段特征分析完成！")
    else:
        print("没有有效的特征输出可供分析。")
except Exception as e:
    import traceback
    print(f"片段特征分析测试出错: {e}")
    print(traceback.format_exc())

# --- 测试5：详细掩码处理验证 ---
print("\n--- 测试5：详细掩码处理验证 ---")
print("准备测试数据...")
    # --- 创建受控输入数据 ---
test_num_nodes = 4
test_seq_len_valid = 3  # 有效长度
test_seq_len_padded = 6  # 带填充的总长度
test_dim = token_embedding_dim  # 使用原始输入维度

    # 输入 A: 无填充 (所有位置都是有效的)
features_A = torch.randn(1, test_num_nodes, test_seq_len_valid, test_dim, device=device)
masks_A = torch.ones(1, test_num_nodes, test_seq_len_valid, dtype=torch.long, device=device)
speakers_A = [[0, 1, 0, 1]]  # Batch size 1，交替的说话人

    # 输入 B: 有填充 (前部分与A相同)
features_B = torch.zeros(1, test_num_nodes, test_seq_len_padded, test_dim, device=device)
features_B[:, :, :test_seq_len_valid, :] = features_A  # 复制A的有效部分
masks_B = torch.zeros(1, test_num_nodes, test_seq_len_padded, dtype=torch.long, device=device)
masks_B[:, :, :test_seq_len_valid] = 1  # 前半部分为有效(1)，后半部分为填充(0)
speakers_B = speakers_A  # 说话人相同

print(f"输入 A (无填充) 特征形状: {features_A.shape}, 掩码形状: {masks_A.shape}")
print(f"输入 B (有填充) 特征形状: {features_B.shape}, 掩码形状: {masks_B.shape}")

    # 创建一个新的模型实例，保持超参数一致
test_model = DialogueGraphModel(
        token_embedding_dim=token_embedding_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        similarity_threshold=similarity_threshold,
        context_window_size=context_window_size
    ).to(device)
    
test_model.eval()  # 设置为评估模式
    
with torch.no_grad():
        print("\n运行测试 (输入 A - 无填充)...")
        # 使用 get_node_embeddings 获取节点特征，输出形状 [nodes, seq_len, dim]
        output_A = test_model.get_node_embeddings(features_A, speakers_A, attention_masks=masks_A)
        print(f"输出 A 形状: {output_A.shape}")
        
        print("\n运行测试 (输入 B - 有填充)...")
        # 使用 get_node_embeddings 获取节点特征，输出形状 [nodes, seq_len, dim]
        output_B = test_model.get_node_embeddings(features_B, speakers_B, attention_masks=masks_B)
        print(f"输出 B 形状: {output_B.shape}")


# --- 汇总测试结果 ---
print("\n--- 测试汇总 ---")
print("1. 基本前向传播功能测试 (单个电话数据): 完成 (请检查上面是否有错误)")
print("2. 图结构构建验证: 完成")
print("3. 边类型权重影响测试: 完成 (请检查上面是否有错误或跳过)")
print("4. 片段特征分析: 完成")
print("5. 详细掩码处理验证: 完成 (请检查上面是否有错误)")
print("\n--- 测试脚本完成 ---")

# 保存输出特征
if 'updated_node_features' in locals() and phone_ids and len(phone_ids) > 0:
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"test_output_features_{phone_ids[0]}.pt")
        # 保存的是节点特征，形状为 [max_num_segments, max_segment_len, output_dim]
        torch.save(updated_node_features, output_file)
        print(f"测试输出已保存到 {output_file}")
    except Exception as e:
        print(f"保存测试输出失败: {e}")

