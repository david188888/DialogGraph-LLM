import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Node:
    """节点类，表示对话中的一个话语"""
    
    def __init__(self, node_id, utterance_features, speaker_id, timestamp=None):
        """
        初始化节点
        
        参数:
            node_id: 节点唯一标识符
            utterance_features: 话语的多模态融合特征 (可以是一个 (token_num, dim) 的张量)
            speaker_id: 说话者ID
            timestamp: 时间戳，用于保持时序信息
        """
        self.id = node_id
        self.utterance_features = utterance_features  # 原始特征，可能是 (token_num, dim)
        self.aggregated_feature = None # 聚合后的固定维度特征
        self.speaker_id = speaker_id
        self.timestamp = timestamp
        self.embedding = None  # GAT 输出的最终节点嵌入
    
    def aggregate_features(self, method='mean'):
        """将原始特征聚合为固定维度向量"""
        if isinstance(self.utterance_features, torch.Tensor) and self.utterance_features.ndim == 2:
            if method == 'mean':
                self.aggregated_feature = torch.mean(self.utterance_features, dim=0)
            elif method == 'max':
                self.aggregated_feature = torch.max(self.utterance_features, dim=0)[0]
            else:
                 # Default to mean pooling if method unknown or feature is already 1D
                 self.aggregated_feature = torch.mean(self.utterance_features, dim=0)
        elif isinstance(self.utterance_features, torch.Tensor) and self.utterance_features.ndim == 1:
             self.aggregated_feature = self.utterance_features # Already aggregated
        else:
            # Handle cases where aggregation is not applicable or needed
             self.aggregated_feature = self.utterance_features # Assume it's pre-aggregated

    def set_embedding(self, embedding):
        """设置节点的最终嵌入表示"""
        self.embedding = embedding
        
    def get_aggregated_feature(self):
        """获取聚合后的节点特征"""
        if self.aggregated_feature is None:
            self.aggregate_features() # Perform default aggregation if not done yet
        return self.aggregated_feature

    def get_embedding(self):
        """获取节点的嵌入表示"""
        return self.embedding


class Edge:
    """边类，表示节点之间的关系"""
    
    # 边的类型ID
    TEMPORAL_EDGE = 1  # 历史时序边
    USER_CONTEXT_EDGE = 2  # 用户关联边
    CROSS_TURN_EDGE = 3  # 跨轮次关联边
    SELF_LOOP_EDGE = 4  # 自环边
    
    def __init__(self, source_id, target_id, edge_type, weight=None):
        """
        初始化边
        
        参数:
            source_id: 源节点ID
            target_id: 目标节点ID
            edge_type: 边的类型 (1-4)
            weight: 边的权重，默认为None，后续通过注意力机制计算
        """
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
    
    def set_weight(self, weight):
        """设置边的权重"""
        self.weight = weight
        
    def get_type_name(self):
        """获取边类型的名称"""
        if self.edge_type == self.TEMPORAL_EDGE:
            return "时序边"
        elif self.edge_type == self.USER_CONTEXT_EDGE:
            return "用户关联边"
        elif self.edge_type == self.CROSS_TURN_EDGE:
            return "跨轮次关联边"
        elif self.edge_type == self.SELF_LOOP_EDGE:
            return "自环边"
        else:
            return "未知边类型"


class SpeakerEmbedding(nn.Module):
    """说话者嵌入模块，为每个说话者构建可学习的嵌入向量"""
    
    def __init__(self, embedding_dim, num_speakers=None):
        """
        初始化说话者嵌入模块
        
        参数:
            embedding_dim: 嵌入维度，将与节点特征维度相同
            num_speakers: 可选，说话者数量；若未指定则动态处理
        """
        super(SpeakerEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        
        # 动态说话人映射字典
        self.speaker_to_idx = {}
        self.next_idx = 0
        
        # 使用嵌入层而非线性层，以便处理变长说话人ID
        if num_speakers is not None:
            self.embedding = nn.Embedding(num_speakers, embedding_dim)
        else:
            # 初始容量为2，后续动态扩展
            self.embedding = nn.Embedding(2, embedding_dim)
    
    def _expand_embedding_if_needed(self, max_id):
        """根据需要扩展嵌入层的容量"""
        current_size = self.embedding.num_embeddings
        if max_id >= current_size:
            # 创建更大容量的新嵌入层
            new_size = max(max_id + 1, current_size * 2)  # 至少扩大一倍
            new_embedding = nn.Embedding(new_size, self.embedding_dim)
            
            # 复制旧权重
            with torch.no_grad():
                new_embedding.weight[:current_size] = self.embedding.weight
            
            # 替换旧嵌入层
            self.embedding = new_embedding.to(self.embedding.weight.device)
            
    def forward(self, speaker_ids):
        """
        计算说话者嵌入
        
        参数:
            speaker_ids: 说话者ID列表或张量 [N]
            
        返回:
            说话者嵌入向量 [N, embedding_dim]
        """
        # 确保输入在正确的设备上
        device = self.embedding.weight.device
        
        # 处理列表输入
        if isinstance(speaker_ids, list):
            # 动态映射说话人ID
            if self.num_speakers is None:
                indices = []
                for speaker_id in speaker_ids:
                    if speaker_id not in self.speaker_to_idx:
                        self.speaker_to_idx[speaker_id] = self.next_idx
                        self.next_idx += 1
                    indices.append(self.speaker_to_idx[speaker_id])
                
                # 检查是否需要扩展嵌入层
                max_idx = max(indices) if indices else -1
                if max_idx >= self.embedding.num_embeddings:
                    self._expand_embedding_if_needed(max_idx)
                
                # 转换为张量
                speaker_indices = torch.tensor(indices, dtype=torch.long, device=device)
            else:
                # 固定数量说话人，直接使用ID作为索引
                speaker_indices = torch.tensor(speaker_ids, dtype=torch.long, device=device)
                
        # 处理张量输入
        elif isinstance(speaker_ids, torch.Tensor):
            speaker_ids = speaker_ids.to(device)
            
            # 动态映射
            if self.num_speakers is None:
                unique_ids = speaker_ids.unique().cpu().tolist()
                for speaker_id in unique_ids:
                    if speaker_id not in self.speaker_to_idx:
                        self.speaker_to_idx[speaker_id] = self.next_idx
                        self.next_idx += 1
                
                # 映射ID到索引
                indices = [self.speaker_to_idx[sid.item()] for sid in speaker_ids]
                max_idx = max(indices) if indices else -1
                if max_idx >= self.embedding.num_embeddings:
                    self._expand_embedding_if_needed(max_idx)
                
                speaker_indices = torch.tensor(indices, dtype=torch.long, device=device)
            else:
                # 固定数量说话人，直接使用
                speaker_indices = speaker_ids.long()
        else:
            raise ValueError(f"不支持的speaker_ids类型: {type(speaker_ids)}")
        
        # 获取嵌入向量
        return self.embedding(speaker_indices)


class DialogueGraph:
    """对话图，表示一个完整对话的图结构"""
    
    def __init__(self):
        """初始化对话图"""
        self.nodes = {}  # 节点字典，键为节点ID，值为Node对象
        self.edges = []  # 边列表
        self.edge_set = set()  # 用于快速检查边是否存在的集合
        # Note: graph tensors (node_features, edge_index, edge_type) are now built dynamically in the model's forward pass
        
    def add_node(self, node):
        """添加节点到图中"""
        self.nodes[node.id] = node
        
    def add_edge(self, edge):
        """添加边到图中，如果边已存在则跳过"""
        # 创建边的唯一标识符(源节点ID, 目标节点ID, 边类型)
        edge_key = (edge.source_id, edge.target_id, edge.edge_type)
        
        # 检查边是否已存在
        if edge_key not in self.edge_set:
            self.edges.append(edge)
            self.edge_set.add(edge_key)
        
    def build_temporal_edges(self):
        """
        构建历史时序边，连接节点i到节点i+1
        所有边都是有向边，从历史节点指向未来节点
        使用更新后的add_edge方法避免创建重复的边
        """
        nodes_list = sorted(self.nodes.values(), key=lambda x: x.timestamp if x.timestamp is not None else x.id)
        for i in range(len(nodes_list) - 1):
            # i -> i+1，从过去指向未来
            edge = Edge(nodes_list[i].id, nodes_list[i+1].id, Edge.TEMPORAL_EDGE)
            self.add_edge(edge)
            
    def build_user_context_edges(self, window_size=4):
        """
        构建用户关联边 (类型 2)
        针对每个用户话语节点，我们构建了一个大小为 k 的过去节点的上下文窗口，
        对这个窗口中的历史用户话语节点都构建一条边指向当前用户话语节点。
        说白了：寻找时间上最近的k个同说话人的历史节点，建立从它们到当前节点的边。
        
        参数:
            window_size: 上下文窗口大小
        """
        # 按时间戳排序节点
        nodes_list = sorted(self.nodes.values(), key=lambda x: x.timestamp if x.timestamp is not None else x.id)
        
        # 遍历所有节点
        for i in range(1, len(nodes_list)):  # 从第二个节点开始，因为第一个没有历史节点
            current_node = nodes_list[i]
            current_speaker = current_node.speaker_id
            
            # 记录已找到的相同说话者的节点数量
            found_same_speaker = 0
            
            # 向前查找window_size个相同说话者的节点
            for j in range(i-1, -1, -1):  # 从i-1向前遍历
                prev_node = nodes_list[j]
                
                # 如果找到相同说话者的节点
                if prev_node.speaker_id == current_speaker:
                    # 建立一条从历史节点到当前节点的边
                    edge = Edge(prev_node.id, current_node.id, Edge.USER_CONTEXT_EDGE)
                    self.add_edge(edge)
                    
                    found_same_speaker += 1
                    
                    # 如果已找到window_size个相同说话者的节点，停止查找
                    if found_same_speaker >= window_size:
                        break

    def build_cross_turn_edges(self, similarity_matrix, threshold):
        """
        构建跨轮次关联边 (类型 3)
        原始定义：用户话语节点i连接到之前的客服或用户话语节点j (j < i)，如果相似度 > threshold，则 j -> i。
        
        参数:
            similarity_matrix: 节点间（使用聚合特征计算）相似度矩阵 [N, N]
            threshold: 相似度阈值
        """
        nodes_list = sorted(self.nodes.values(), key=lambda x: x.timestamp if x.timestamp is not None else x.id)
        node_ids = [node.id for node in nodes_list]
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        num_nodes = len(nodes_list)
        for i in range(num_nodes): # 当前节点 i
            # TODO: Add logic to check if nodes_list[i] is a user node if needed
            
            for j in range(i): # 之前的节点 j
                # 获取节点在矩阵中的索引
                idx_i = node_id_to_idx[nodes_list[i].id]
                idx_j = node_id_to_idx[nodes_list[j].id]
                
                # 如果相似度大于阈值，则构建边 j -> i
                if similarity_matrix[idx_j, idx_i] > threshold:
                    # 源节点 j 可以是用户或客服
                    edge = Edge(nodes_list[j].id, nodes_list[i].id, Edge.CROSS_TURN_EDGE)
                    self.add_edge(edge)
                    # # (可选) 添加反向边 i -> j
                    # edge_rev = Edge(nodes_list[i].id, nodes_list[j].id, Edge.CROSS_TURN_EDGE)
                    # self.add_edge(edge_rev)

    def build_self_loop_edges(self):
        """
        构建自环边 (类型 4)
        """
        for node_id in self.nodes:
            edge = Edge(node_id, node_id, Edge.SELF_LOOP_EDGE)
            self.add_edge(edge)
            
    def get_tensors(self, device):
        """
        提取图结构的边信息为张量，特征聚合在模型 forward 中完成。
        
        返回:
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            node_ids_map: 节点原始ID到其在排序后列表中的索引的映射
            sorted_node_ids: 排序后的节点ID列表
        """
        if not self.nodes:
            # Handle empty graph case
            return (torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.long, device=device),
                    {}, [])

        # 按照 timestamp (或 ID) 排序节点，以确保一致性
        sorted_nodes = sorted(self.nodes.values(), key=lambda node: node.timestamp if node.timestamp is not None else node.id)
        sorted_node_ids = [node.id for node in sorted_nodes]
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}

        edge_index = []
        edge_type = []
        for edge in self.edges:
            # 获取节点在排序后列表中的索引
            source_idx = node_id_to_idx.get(edge.source_id)
            target_idx = node_id_to_idx.get(edge.target_id)

            # Ensure both nodes exist in the sorted list
            if source_idx is not None and target_idx is not None:
                edge_index.append([source_idx, target_idx])
                edge_type.append(edge.edge_type)
            else:
                print(f"Warning: Edge ({edge.source_id} -> {edge.target_id}) skipped because one or both nodes not found in the graph's node list.")


        if edge_index:
             edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()  # [2, num_edges]
        else:
             edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)  # [num_edges]

        return edge_index, edge_type, node_id_to_idx, sorted_node_ids


class MultiHeadAttention(nn.Module):
    """使用PyTorch官方nn.MultiheadAttention实现的图注意力机制"""
    
    def __init__(self, in_features, out_features, num_heads, dropout=0.2, num_edge_types=4):
        """
        初始化多头注意力模块
        
        参数:
            in_features: 输入特征维度 (聚合后的维度)
            out_features: 输出特征维度 (需要能被 num_heads 整除)
            num_heads: 注意力头数量
            dropout: Dropout概率
            num_edge_types: 边类型的数量，默认为4
        """
        super(MultiHeadAttention, self).__init__()
        
        # 确保out_features可被num_heads整除
        if out_features % num_heads != 0:
            print(f"Warning: out_features ({out_features}) not divisible by num_heads ({num_heads}). Adjusting out_features.")
            out_features = (out_features // num_heads) * num_heads
            if out_features == 0:
                raise ValueError("out_features becomes 0 after adjustment.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        
        # 为每种边类型添加可学习的权重参数
        # 初始化为全1，表示所有边类型初始时具有相同的重要性
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types))
        
        # 为保持与GAT兼容性，添加投影层
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        
        # 使用PyTorch官方的多头注意力实现
        self.attn = nn.MultiheadAttention(
            embed_dim=out_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout层用于输出
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_type=None):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵 [num_nodes, in_features] (聚合后的特征)
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]，如果为None则所有边被视为相同类型
            
        返回:
            输出特征矩阵 [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        if num_nodes == 0 or edge_index.numel() == 0:
            # 处理空图或无边的情况
            return torch.zeros(num_nodes, self.out_features, device=x.device)
            
        # 投影查询、键和值
        query = self.q_proj(x).unsqueeze(0)  # [1, num_nodes, out_features]
        key = self.k_proj(x).unsqueeze(0)    # [1, num_nodes, out_features]
        value = self.v_proj(x).unsqueeze(0)  # [1, num_nodes, out_features]
        
        # 构建基础注意力掩码（所有位置初始化为-inf，表示不允许注意力）
        attn_mask = torch.full((num_nodes, num_nodes), float('-inf'), device=x.device)
        
        src_nodes, dst_nodes = edge_index
        
        if edge_type is not None:
            # 使用边类型权重
            for i in range(edge_index.shape[1]):
                src, dst = src_nodes[i], dst_nodes[i]
                # 获取边类型（从1开始），减1作为索引（从0开始）
                e_type_idx = edge_type[i].item() - 1
                # 使用softplus确保权重为正数，更稳定地学习
                weight = F.softplus(self.edge_type_weights[e_type_idx])
                # 应用边权重作为初始注意力得分
                attn_mask[src, dst] = weight
        else:
            # 如果没有提供边类型，所有边使用相同的权重（设为1，即log(e)）
            attn_mask[src_nodes, dst_nodes] = 1.0
            
        # 添加自环（如果未在edge_index中明确包含）
        for i in range(num_nodes):
            if attn_mask[i, i] == float('-inf'):  # 确保自环不会覆盖已有的边
                # 如果提供了边类型，使用自环边类型权重（通常是类型4）
                if edge_type is not None:
                    self_loop_type_idx = 3  # 自环边是类型4，索引为3
                    weight = F.softplus(self.edge_type_weights[self_loop_type_idx])
                    attn_mask[i, i] = weight
                else:
                    attn_mask[i, i] = 1.0
            
        # 应用多头注意力
        output, _ = self.attn(
            query=query,           # [1, num_nodes, out_features]
            key=key,               # [1, num_nodes, out_features]
            value=value,           # [1, num_nodes, out_features]
            attn_mask=attn_mask,   # [num_nodes, num_nodes]
            need_weights=False     # 不需要返回注意力权重
        )
        
        # 删除批次维度并应用dropout
        output = output.squeeze(0)  # [num_nodes, out_features]
        output = self.dropout_layer(output)
        
        return output


class DebugMultiHeadAttention(nn.Module):
    """调试版本的多头注意力机制，返回注意力权重用于分析"""
    
    def __init__(self, in_features, out_features, num_heads, dropout=0.2, num_edge_types=4):
        """初始化参数与原版保持一致"""
        super(DebugMultiHeadAttention, self).__init__()
        
        # 确保out_features可被num_heads整除
        if out_features % num_heads != 0:
            print(f"Warning: out_features ({out_features}) not divisible by num_heads ({num_heads}). Adjusting out_features.")
            out_features = (out_features // num_heads) * num_heads
            if out_features == 0:
                raise ValueError("out_features becomes 0 after adjustment.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        
        # 为每种边类型添加可学习的权重参数
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types))
        
        # 投影层
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        
        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=out_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_type=None, return_attention=True):
        """
        前向传播，返回节点表示和注意力权重
        
        参数:
            x: 节点特征矩阵 [num_nodes, in_features]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            return_attention: 是否返回注意力权重
            
        返回:
            output: 输出特征矩阵 [num_nodes, out_features]
            attn_weights: 注意力权重 [batch_size, num_nodes, num_nodes]
            edge_weights: 边类型权重 [num_edge_types]
            attn_mask: 注意力掩码矩阵 [num_nodes, num_nodes]
        """
        num_nodes = x.size(0)
        if num_nodes == 0 or edge_index.numel() == 0:
            if return_attention:
                return (torch.zeros(num_nodes, self.out_features, device=x.device), 
                        None, self.edge_type_weights, None)
            else:
                return torch.zeros(num_nodes, self.out_features, device=x.device)
            
        # 投影查询、键和值
        query = self.q_proj(x).unsqueeze(0)  # [1, num_nodes, out_features]
        key = self.k_proj(x).unsqueeze(0)    # [1, num_nodes, out_features]
        value = self.v_proj(x).unsqueeze(0)  # [1, num_nodes, out_features]
        
        # 构建注意力掩码
        attn_mask = torch.full((num_nodes, num_nodes), float('-inf'), device=x.device)
        
        src_nodes, dst_nodes = edge_index
        
        # 创建一个字典记录边类型到权重的映射，用于调试
        edge_type_to_weight = {}
        
        if edge_type is not None:
            # 使用边类型权重
            for i in range(edge_index.shape[1]):
                src, dst = src_nodes[i], dst_nodes[i]
                e_type_idx = edge_type[i].item() - 1
                weight = F.softplus(self.edge_type_weights[e_type_idx])
                attn_mask[src, dst] = weight
                
                # 记录边类型到权重的映射
                if e_type_idx not in edge_type_to_weight:
                    edge_type_to_weight[e_type_idx] = weight.item()
        else:
            attn_mask[src_nodes, dst_nodes] = 1.0
            
        # 添加自环
        for i in range(num_nodes):
            if attn_mask[i, i] == float('-inf'):
                if edge_type is not None:
                    self_loop_type_idx = 3  # 自环边类型索引
                    weight = F.softplus(self.edge_type_weights[self_loop_type_idx])
                    attn_mask[i, i] = weight
                    
                    # 记录自环边权重
                    if self_loop_type_idx not in edge_type_to_weight:
                        edge_type_to_weight[self_loop_type_idx] = weight.item()
                else:
                    attn_mask[i, i] = 1.0
            
        # 应用多头注意力，返回注意力权重
        output, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            need_weights=return_attention
        )
        
        output = output.squeeze(0)
        output = self.dropout_layer(output)
        
        if return_attention:
            edge_weights = {t+1: edge_type_to_weight[t] for t in edge_type_to_weight}
            return output, attn_weights, edge_weights, attn_mask
        else:
            return output


class GraphAttentionNetwork(nn.Module):
    """图注意力网络，用于对话图的节点表示学习"""
    
    def __init__(self, aggregated_input_dim, hidden_dim, output_dim, num_heads,
                 speaker_embedding_dim=None, num_speakers=None, num_layers=2, dropout=0.2):
        """
        初始化图注意力网络
        
        参数:
            aggregated_input_dim: 聚合后的节点特征维度
            hidden_dim: GAT 隐藏层维度 (注意：要能被 num_heads 整除)
            output_dim: GAT 输出特征维度 (注意：要能被 num_heads 整除)
            num_heads: 注意力头数量
            speaker_embedding_dim: 说话者嵌入维度，若为None则设为等于aggregated_input_dim
            num_speakers: 说话者数量，None表示动态确定
            num_layers: 图注意力层数量
            dropout: Dropout概率
        """
        super(GraphAttentionNetwork, self).__init__()
        # Ensure hidden_dim and output_dim are divisible by num_heads for concatenation
        if hidden_dim % num_heads != 0:
             print(f"Warning: hidden_dim ({hidden_dim}) not divisible by num_heads ({num_heads}). Adjusting hidden_dim.")
             hidden_dim = (hidden_dim // num_heads) * num_heads
             if hidden_dim == 0: raise ValueError("hidden_dim becomes 0 after adjustment.")
        if num_layers > 1 and output_dim % num_heads != 0: # Output layer might aggregate differently if num_layers=1
             print(f"Warning: output_dim ({output_dim}) not divisible by num_heads ({num_heads}). Adjusting output_dim.")
             output_dim = (output_dim // num_heads) * num_heads
             if output_dim == 0: raise ValueError("output_dim becomes 0 after adjustment.")
        elif num_layers == 1 and output_dim % num_heads != 0:
            # If only one layer, the output layer IS the first layer.
            # We might allow aggregation (like mean) instead of concat here, or force divisibility.
            # Forcing divisibility for simplicity.
            print(f"Warning: output_dim ({output_dim}) not divisible by num_heads ({num_heads}) for single GAT layer. Adjusting output_dim.")
            output_dim = (output_dim // num_heads) * num_heads
            if output_dim == 0: raise ValueError("output_dim becomes 0 after adjustment.")


        self.aggregated_input_dim = aggregated_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_edge_types = 4 # Fixed based on description
        
        # 若未指定speaker_embedding_dim，则设为与节点特征维度相同
        if speaker_embedding_dim is None:
            speaker_embedding_dim = aggregated_input_dim

        # 说话者嵌入层
        self.speaker_embedding = SpeakerEmbedding(speaker_embedding_dim, num_speakers)
        
        # 注意：现在说话人嵌入维度与节点特征维度相同，因此传入投影层的是双倍维度
        combined_dim = aggregated_input_dim + speaker_embedding_dim

        # 输入特征投影
        self.input_projection = nn.Linear(combined_dim, hidden_dim)

        # 图注意力层
        self.gat_layers = nn.ModuleList()
        if num_layers == 1:
             # Directly map projected input to output
             self.gat_layers.append(
                 MultiHeadAttention(hidden_dim, output_dim, num_heads, dropout) # Pass dropout to MultiHeadAttention
             )
        else:
            # First layer: projected_dim -> hidden_dim
            self.gat_layers.append(
                MultiHeadAttention(hidden_dim, hidden_dim, num_heads, dropout)
            )
            # Intermediate layers: hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                self.gat_layers.append(
                    MultiHeadAttention(hidden_dim, hidden_dim, num_heads, dropout)
                )
            # Output layer: hidden_dim -> output_dim
            self.gat_layers.append(
                MultiHeadAttention(hidden_dim, output_dim, num_heads, dropout)
            )

        self.dropout_layer = nn.Dropout(dropout) # Used between layers
        self.activation = nn.ELU()

    def forward(self, aggregated_x, speaker_ids, edge_index, edge_type=None):
        """
        前向传播
        
        参数:
            aggregated_x: 聚合后的话语特征矩阵 [num_nodes, aggregated_input_dim]
            speaker_ids: 说话者ID [num_nodes]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]，如果为None则所有边被视为相同类型
            
        返回:
            节点的最终表示 [num_nodes, output_dim]
        """
        # 1. 计算说话者嵌入
        speaker_emb = self.speaker_embedding(speaker_ids)  # [num_nodes, speaker_embedding_dim]

        # 2. 拼接聚合特征和说话者嵌入
        x = torch.cat([aggregated_x, speaker_emb], dim=1)  # [num_nodes, aggregated_input_dim + speaker_embedding_dim]

        # 3. 投影到隐藏维度
        x = self.input_projection(x)  # [num_nodes, hidden_dim]
        # Apply activation and dropout after projection, before first GAT layer
        x = self.activation(x)
        x = self.dropout_layer(x)


        # 4. 应用图注意力层
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_type) # 传递边类型
            # Apply activation and dropout *between* GAT layers (not after the last one)
            if i < self.num_layers - 1:
                 x = self.activation(x)
                 x = self.dropout_layer(x)


        return x


class DialogueGraphBuilder:
    """对话图构建器，负责从原始对话数据构建对话图结构（边）"""
    
    def __init__(self, similarity_threshold=0.5, context_window_size=4):
        """
        初始化对话图构建器
        
        参数:
            similarity_threshold: 相似度阈值，用于构建跨轮次关联边
            context_window_size: 上下文窗口大小，用于构建用户关联边
        """
        self.similarity_threshold = similarity_threshold
        self.context_window_size = context_window_size

    def compute_similarity_matrix(self, aggregated_node_features):
        """
        计算节点间的相似度矩阵 (使用聚合后的特征)
        
        参数:
            aggregated_node_features: 聚合后的节点特征张量 [N, agg_dim]
            
        返回:
            相似度矩阵 [N, N]
        """
        if aggregated_node_features is None or aggregated_node_features.nelement() == 0:
             # Return an empty matrix or a matrix of appropriate size with zeros/NaNs
             num_nodes = 0 # Or determine from context if possible
             if aggregated_node_features is not None:
                  num_nodes = aggregated_node_features.shape[0]
             return np.zeros((num_nodes, num_nodes))


        # 归一化特征
        normalized_features = F.normalize(aggregated_node_features, p=2, dim=1)

        # 计算余弦相似度
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())

        # Clamp values to avoid potential numerical issues if needed
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)

        return similarity_matrix.cpu().numpy() # Return numpy array as before

    def build_graph_structure(self, num_nodes, speaker_ids_list, timestamps=None):
        """
        构建图结构 (节点和除跨轮次外的边)
        
        参数:
            num_nodes: 节点数量
            speaker_ids_list: 说话者ID列表 [N]
            timestamps: 时间戳列表 [N] (如果为None则使用索引)
            
        返回:
            构建好的对话图对象 (只包含节点、时序边、用户关联边、自环边)
        """
        graph = DialogueGraph()

        # 创建节点 (此时特征可以是 None 或 原始特征，聚合特征在模型中处理)
        if timestamps is None:
            timestamps = list(range(num_nodes))
        elif len(timestamps) != num_nodes:
             raise ValueError(f"Length of timestamps ({len(timestamps)}) must match num_nodes ({num_nodes})")


        for i in range(num_nodes):
            # utterance_features can be set later or kept as None if only structure is needed here
            node = Node(node_id=i, utterance_features=None, speaker_id=speaker_ids_list[i], timestamp=timestamps[i])
            graph.add_node(node)

        # 构建时序边 (Type 1)
        graph.build_temporal_edges()

        # 构建用户关联边 (Type 2)
        graph.build_user_context_edges(self.context_window_size)

        # 构建自环边 (Type 4)
        graph.build_self_loop_edges()

        # 注意：跨轮次边 (Type 3) 在 DialogueGraphModel.forward 中构建，因为它需要聚合特征

        return graph


class DialogueGraphModel(nn.Module):
    """对话图模型，集成图注意力网络和其他组件"""
    
    def __init__(self, token_embedding_dim, hidden_dim=None, output_dim=None, num_heads=4,
                 speaker_embedding_dim=None, num_speakers=None, num_layers=2, dropout=0.2,
                 similarity_threshold=0.5, context_window_size=3, aggregation_method='mean'):
        """
        初始化对话图模型
        
        参数:
            token_embedding_dim: 输入的 token 嵌入维度 (聚合后的维度将与此相同)
            hidden_dim: GAT 隐藏层维度, 如果为None则设置为 2*token_embedding_dim
            output_dim: GAT 输出特征维度, 如果为None则设置为等于 token_embedding_dim 以保持维度不变
            num_heads: 注意力头数量
            speaker_embedding_dim: 说话者嵌入维度，若为None则设为等于token_embedding_dim
            num_speakers: 说话者数量，None表示动态确定
            num_layers: 图注意力层数量
            dropout: Dropout概率
            similarity_threshold: 相似度阈值 (初始值，可以设为可学习)
            context_window_size: 上下文窗口大小
            aggregation_method: 'mean' or 'max' for aggregating token embeddings
        """
        super(DialogueGraphModel, self).__init__()

        self.aggregation_method = aggregation_method
        self.token_embedding_dim = token_embedding_dim
        
        # 如果未指定hidden_dim，默认设为token_embedding_dim的2倍
        if hidden_dim is None:
            hidden_dim = 2 * token_embedding_dim
            
        # 如果未指定output_dim，默认设为等于token_embedding_dim
        if output_dim is None:
            output_dim = token_embedding_dim
            
        # 如果未指定speaker_embedding_dim，默认设为等于token_embedding_dim
        if speaker_embedding_dim is None:
            speaker_embedding_dim = token_embedding_dim
            
        # 确保hidden_dim和output_dim可以被num_heads整除
        if hidden_dim % num_heads != 0:
            print(f"Warning: Adjusting hidden_dim from {hidden_dim} to be divisible by num_heads ({num_heads})")
            hidden_dim = (hidden_dim // num_heads) * num_heads
            
        if output_dim % num_heads != 0:
            print(f"Warning: Adjusting output_dim from {output_dim} to be divisible by num_heads ({num_heads})")
            output_dim = (output_dim // num_heads) * num_heads
            
        # 确保调整后的dimensions不为0
        if hidden_dim == 0:
            hidden_dim = num_heads  # 最小可能值
            print(f"Warning: hidden_dim was adjusted to minimum value: {hidden_dim}")
            
        if output_dim == 0:
            output_dim = num_heads  # 最小可能值
            print(f"Warning: output_dim was adjusted to minimum value: {output_dim}")

        # 图注意力网络 (输入维度是 token_embedding_dim)
        self.gat = GraphAttentionNetwork(
            aggregated_input_dim=token_embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            speaker_embedding_dim=speaker_embedding_dim,
            num_speakers=num_speakers,
            num_layers=num_layers,
            dropout=dropout
        )

        # 对话图构建器 (用于构图逻辑)
        self.graph_builder = DialogueGraphBuilder(
            similarity_threshold=similarity_threshold, # Initial value
            context_window_size=context_window_size
        )

        # 可学习的相似度阈值 (可选)
        # self.similarity_threshold = nn.Parameter(torch.tensor(similarity_threshold))
        self.similarity_threshold_value = similarity_threshold # Use fixed threshold for now
        
        # 存储初始化参数，方便使用
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def _aggregate_features(self, features_list, device):
        """Helper function to aggregate list of (tok_num, dim) tensors."""
        aggregated = []
        if not features_list: # Handle empty input list
            return torch.empty((0, self.token_embedding_dim), device=device)

        for i, feat in enumerate(features_list):
            if isinstance(feat, torch.Tensor):
                feat = feat.to(device) # Move to device
                if feat.ndim == 2 and feat.shape[0] > 0: # Check if it's (token_num, dim) and not empty
                    if self.aggregation_method == 'mean':
                        agg = torch.mean(feat, dim=0)
                    elif self.aggregation_method == 'max':
                        agg = torch.max(feat, dim=0)[0]
                    else:
                        agg = torch.mean(feat, dim=0) # Default
                    aggregated.append(agg)
                elif feat.ndim == 1: # Assume already aggregated
                     if feat.shape[0] == self.token_embedding_dim:
                         aggregated.append(feat)
                     else:
                         print(f"Warning: Node {i} feature tensor is 1D but has wrong dimension ({feat.shape[0]} vs {self.token_embedding_dim}). Using zero vector.")
                         aggregated.append(torch.zeros(self.token_embedding_dim, device=device))
                else: # Handle empty tensor (shape[0]==0) or unexpected dims
                     aggregated.append(torch.zeros(self.token_embedding_dim, device=device))
                     print(f"Warning: Node {i} feature tensor had unexpected shape {feat.shape} or was empty. Using zero vector.")
            else:
                 # Handle non-tensor input
                 aggregated.append(torch.zeros(self.token_embedding_dim, device=device))
                 print(f"Warning: Node {i} feature was not a tensor ({type(feat)}). Using zero vector.")

        # Stack the aggregated features
        if not aggregated: # Should not happen if features_list is not empty, but as safeguard
             return torch.empty((0, self.token_embedding_dim), device=device)

        return torch.stack(aggregated) # [num_nodes, token_embedding_dim]

    def forward(self, utterance_features_input, speaker_ids_input, edge_index=None, edge_type=None):
        """
        前向传播
        
        参数:
            utterance_features_input:
                - List of Tensors: [(token_num_1, dim), (token_num_2, dim), ...] (for batch_size=1, auto-graph build)
                - Tensor: [batch_size * seq_len, dim] (if features are pre-aggregated and graph is provided)
            speaker_ids_input:
                - List or Tensor: [seq_len] (for batch_size=1, auto-graph build)
                - Tensor: [batch_size * seq_len] (if graph is provided)
            edge_index: 预定义的边索引 [2, num_edges], 如果为None则自动构建图 (仅支持 batch_size=1)
            edge_type: 预定义的边类型 [num_edges], 如果为None则自动构建图 (仅支持 batch_size=1)
            
        返回:
            节点 GAT 输出嵌入:
                - [1, seq_len, output_dim] (if batch_size=1, auto-graph build)
                - [N, output_dim] (if graph is provided, N = batch_size * seq_len, needs reshaping by caller)
        """
        device = next(self.parameters()).device # Get model's device

        # --- Case 1: Auto-build graph (expects batch_size=1 and list input) ---
        if edge_index is None or edge_type is None:
            # --- Input Validation ---
            if not isinstance(utterance_features_input, list):
                raise ValueError("For automatic graph building (edge_index is None), utterance_features_input must be a list of tensors.")
            if isinstance(speaker_ids_input, torch.Tensor):
                if speaker_ids_input.ndim > 1 or speaker_ids_input.shape[0] != len(utterance_features_input):
                    raise ValueError(f"Speaker_ids tensor shape ({speaker_ids_input.shape}) incompatible with feature list length ({len(utterance_features_input)}). Expected [seq_len].")
                speaker_ids_list_int = speaker_ids_input.cpu().tolist() # For graph builder - 不转换为long
                speaker_ids_tensor = speaker_ids_input.to(device) # For GAT - 不转换为long
            elif isinstance(speaker_ids_input, list):
                 if len(speaker_ids_input) != len(utterance_features_input):
                     raise ValueError(f"speaker_ids list length ({len(speaker_ids_input)}) must match utterance_features list length ({len(utterance_features_input)}).")
                 speaker_ids_list_int = speaker_ids_input # For graph builder
                 speaker_ids_tensor = torch.tensor(speaker_ids_input, device=device) # For GAT - 不限制dtype
            else:
                raise TypeError(f"speaker_ids_input must be a list or tensor, got {type(speaker_ids_input)}")


            seq_len = len(utterance_features_input)
            if seq_len == 0:
                 return torch.empty((1, 0, self.output_dim), device=device) # Handle empty sequence

            # --- Processing ---
            # 1. Aggregate Features: List[(tok, dim)] -> Tensor[seq_len, dim]
            aggregated_features = self._aggregate_features(utterance_features_input, device) # [seq_len, token_embedding_dim]

            # 2. Build Graph Structure (Types 1, 2, 4)
            graph = self.graph_builder.build_graph_structure(
                num_nodes=seq_len,
                speaker_ids_list=speaker_ids_list_int, # Builder expects list of ints
                timestamps=list(range(seq_len)) # Use index as timestamp
            )

            # 3. Compute Similarity and Build Cross-Turn Edges (Type 3)
            if seq_len > 1:
                similarity_matrix = self.graph_builder.compute_similarity_matrix(aggregated_features) # Use aggregated features
                # Use fixed threshold for now
                graph.build_cross_turn_edges(similarity_matrix, self.similarity_threshold_value)

            # 4. Get Graph Tensors (Edges)
            edge_index, edge_type, _, _ = graph.get_tensors(device) # 获取边类型，不再只是未使用的参数
            
            # 5. Apply Graph Attention Network
            # Input: aggregated_features, speaker_ids_tensor, edge_index, edge_type
            node_embeddings = self.gat(aggregated_features, speaker_ids_tensor, edge_index, edge_type) # 传递边类型

            # Reshape for consistent batch output: [1, seq_len, output_dim]
            node_embeddings = node_embeddings.unsqueeze(0)

        # --- Case 2: Use pre-defined graph ---
        else:
            # --- Input Validation ---
            if not isinstance(utterance_features_input, torch.Tensor):
                 raise ValueError("If edge_index is provided, utterance_features_input must be a pre-aggregated tensor [N, dim].")
            if not isinstance(speaker_ids_input, torch.Tensor):
                 raise ValueError("If edge_index is provided, speaker_ids_input must be a tensor [N].")
            if utterance_features_input.ndim != 2 or utterance_features_input.shape[1] != self.token_embedding_dim:
                raise ValueError(f"Expected pre-aggregated features of shape [N, {self.token_embedding_dim}], got {utterance_features_input.shape}")
            if speaker_ids_input.ndim > 1 or speaker_ids_input.shape[0] != utterance_features_input.shape[0]:
                 raise ValueError(f"Expected speaker_ids of shape [N={utterance_features_input.shape[0]}], got {speaker_ids_input.shape}")


            # --- Processing ---
            aggregated_features = utterance_features_input.to(device)
            speaker_ids_tensor = speaker_ids_input.to(device) # 不转换为long
            edge_index = edge_index.to(device)
            # 确保边类型被传递到设备
            if edge_type is not None:
                edge_type = edge_type.to(device)

            # Apply Graph Attention Network
            node_embeddings = self.gat(aggregated_features, speaker_ids_tensor, edge_index, edge_type) # 传递边类型
            # Caller is responsible for reshaping if needed (e.g., back to [batch, seq, dim])
        
        return node_embeddings
    
    # get_node_embeddings remains similar to the forward pass logic for auto-build case
    def get_node_embeddings(self, utterance_features_list, speaker_ids_list):
        """
        获取对话中所有节点的嵌入表示 (non-batched, builds graph internally).
        
        参数:
            utterance_features_list: 话语特征列表 List[(token_num, dim)]
            speaker_ids_list: 说话者ID列表 List[int]
            
        返回:
            节点嵌入列表 List[Tensor(output_dim)]
        """
        self.eval() # Set to evaluation mode
        device = next(self.parameters()).device
        with torch.no_grad():
             # Ensure inputs are lists
             if not isinstance(utterance_features_list, list):
                 raise TypeError("utterance_features_list must be a list for get_node_embeddings")
             if not isinstance(speaker_ids_list, list):
                 raise TypeError("speaker_ids_list must be a list for get_node_embeddings")

             # Use the forward pass logic for batch_size=1 auto-build
             embeddings_batch = self.forward(utterance_features_list, speaker_ids_list, edge_index=None, edge_type=None) # Output: [1, seq_len, output_dim]

             # Convert back to list
             if embeddings_batch.nelement() > 0:
                 node_embeddings_list = [embeddings_batch[0, i].cpu() for i in range(embeddings_batch.shape[1])] # Move to CPU for list output
             else:
                 node_embeddings_list = []

        return node_embeddings_list