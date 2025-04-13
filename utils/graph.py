import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Node:
    """节点类，表示对话中的一个话语"""
    
    def __init__(self, node_id, utterance_features, speaker_id, timestamp=None, attention_mask=None):
        """
        初始化节点
        
        参数:
            node_id: 节点唯一标识符
            utterance_features: 话语的多模态融合特征 (可以是一个 (token_num, dim) 的张量)
            speaker_id: 说话者ID
            timestamp: 时间戳，用于保持时序信息
            attention_mask: 注意力掩码，标记哪些位置是有效的 (token_num,)
        """
        self.id = node_id
        self.utterance_features = utterance_features  # 原始特征，可能是 (token_num, dim)
        self.speaker_id = speaker_id
        self.timestamp = timestamp
        self.embedding = None  # GAT 输出的最终节点嵌入
        self.attention_mask = attention_mask  # 注意力掩码
    
    def set_embedding(self, embedding):
        """设置节点的最终嵌入表示"""
        self.embedding = embedding
        
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
                    # # # (可选) 添加反向边 i -> j
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


class MultiHeadAttentionWithMask(nn.Module):
    """支持直接处理变长序列特征和掩码的多头注意力机制"""
    
    def __init__(self, dim, num_heads, dropout=0.2, num_edge_types=4):
        """
        初始化支持掩码的多头注意力模块
        
        参数:
            dim: 特征维度 (输入输出维度相同)
            num_heads: 注意力头数量
            dropout: Dropout概率
            num_edge_types: 边类型的数量，默认为4
        """
        super(MultiHeadAttentionWithMask, self).__init__()
        
        # 确保dim可被num_heads整除
        if dim % num_heads != 0:
            print(f"Warning: dim ({dim}) not divisible by num_heads ({num_heads}). Adjusting dim.")
            dim = (dim // num_heads) * num_heads
            if dim == 0:
                raise ValueError("dim becomes 0 after adjustment.")
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        
        # 为每种边类型添加可学习的权重参数
        # 初始化为全1，表示所有边类型初始时具有相同的重要性
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types))
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(dim)
        
        # 投影层 - 保持输入输出维度相同
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, node_features, node_masks, edge_index, edge_type=None):
        """
        前向传播，直接处理变长序列特征和掩码
        
        参数:
            node_features: 节点原始特征列表 [num_nodes, max_seq_len, dim]
            node_masks: 节点特征掩码 [num_nodes, max_seq_len]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]，如果为None则所有边被视为相同类型
            
        返回:
            更新后的节点特征 [num_nodes, max_seq_len, dim]
        """
        num_nodes, max_seq_len, hidden_dim = node_features.size()
        device = node_features.device
        
        if num_nodes == 0 or edge_index.numel() == 0:
            # 处理空图或无边的情况
            return torch.zeros_like(node_features)
        
        # 构建图的邻接矩阵 (带边类型权重)
        adj_matrix = torch.full((num_nodes, num_nodes), float('-inf'), device=device)
        
        src_nodes, dst_nodes = edge_index
        
        if edge_type is not None:
            # 使用边类型权重
            for i in range(edge_index.shape[1]):
                src, dst = src_nodes[i], dst_nodes[i]
                # 获取边类型（从1开始），减1作为索引（从0开始）
                e_type_idx = edge_type[i].item() - 1
                # 使用softplus确保权重为正数，并确保稳定性
                weight = F.softplus(self.edge_type_weights[e_type_idx]).clamp(min=1e-6, max=1e6)
                # 应用边权重作为邻接矩阵中的值
                adj_matrix[src, dst] = weight
        else:
            # 如果没有提供边类型，所有边使用相同的权重
            adj_matrix[src_nodes, dst_nodes] = 1.0
        
        # 应用层归一化
        normed_features = self.layer_norm(node_features)
        
        # 计算查询、键和值
        q = self.q_proj(normed_features)  # [num_nodes, max_seq_len, dim]
        k = self.k_proj(normed_features)  # [num_nodes, max_seq_len, dim]
        v = self.v_proj(normed_features)  # [num_nodes, max_seq_len, dim]
        
        # 重塑为多头形式 - 使用reshape替代view
        head_dim = self.dim // self.num_heads
        q = q.reshape(num_nodes, max_seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [num_nodes, num_heads, max_seq_len, head_dim]
        k = k.reshape(num_nodes, max_seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [num_nodes, num_heads, max_seq_len, head_dim]
        v = v.reshape(num_nodes, max_seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [num_nodes, num_heads, max_seq_len, head_dim]
        
        # 获取每个节点的有效 token 数量 (用于后续计算)
        valid_tokens_per_node = node_masks.sum(dim=1).long()  # [num_nodes]
        
        # 创建输出特征张量，初始化为零
        # 形状: [num_nodes, num_heads, max_seq_len, head_dim]
        output_features = torch.zeros_like(q)
        
        # 对每个节点单独计算 attention
        for i in range(num_nodes):
            # 跳过填充节点 (如果存在)
            if valid_tokens_per_node[i] == 0:
                continue
                
            # 获取当前节点的查询表示和掩码
            # 只保留有效的 token (由掩码确定)
            query_node_mask = node_masks[i]  # [max_seq_len]
            query_valid_len = valid_tokens_per_node[i]
            
            # 对于该节点，收集所有相连节点的信息
            attending_nodes = []
            edge_weights = []
            
            # 找出所有与节点i相连的节点 (边i->j存在)
            for j in range(num_nodes):
                if adj_matrix[i, j] != float('-inf'):
                    attending_nodes.append(j)
                    edge_weights.append(adj_matrix[i, j])
            
            if not attending_nodes:
                continue  # 如果没有相连节点，跳过
                
            # 将列表转换为张量，方便操作
            attending_nodes = torch.tensor(attending_nodes, device=device)
            edge_weights = torch.tensor(edge_weights, device=device)
            
            # 获取所有相连节点的键和值向量
            # 形状: [num_attending, num_heads, max_seq_len, head_dim]
            keys_attending = k[attending_nodes]
            values_attending = v[attending_nodes]
            
            # 获取相连节点的掩码
            # 形状: [num_attending, max_seq_len]
            masks_attending = node_masks[attending_nodes]
            
            # 计算节点i对所有相连节点的注意力分数
            # 为每个头单独计算
            for h in range(self.num_heads):
                # 当前节点在当前头的查询向量 [max_seq_len, head_dim]
                query_head = q[i, h]  
                
                # 仅对有效token计算注意力
                valid_query_head = query_head[:query_valid_len]  # [valid_len, head_dim]
                
                # 存储对所有相连节点的注意力值
                num_attending = len(attending_nodes)
                
                # 对每个相连的节点j
                for idx, j in enumerate(attending_nodes):
                    # 获取节点j的有效token数
                    key_node_mask = masks_attending[idx]  # [max_seq_len]
                    key_valid_len = key_node_mask.sum().long()
                    
                    if key_valid_len == 0:
                        continue  # 跳过没有有效token的节点
                    
                    # 获取当前相连节点的键向量 - 仅有效部分
                    key_head = keys_attending[idx, h]  # [max_seq_len, head_dim]
                    valid_key_head = key_head[:key_valid_len]  # [valid_len_j, head_dim]
                    
                    # 计算该节点对的注意力分数 [valid_len_i, valid_len_j]
                    scores = torch.matmul(valid_query_head, valid_key_head.transpose(0, 1)) / (head_dim ** 0.5)
                    
                    # 加上边权重 (为所有score加上相同的偏置)
                    scores = scores + edge_weights[idx]
                    
                    # 对每个查询位置的分数进行softmax - 关键修改点!
                    # 这样每个有效的查询token只会关注有效的键token
                    attn_weights = F.softmax(scores, dim=1)  # [valid_len_i, valid_len_j]
                    
                    # 对噪声处理
                    attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 获取当前相连节点的值向量 - 仅有效部分
                    value_head = values_attending[idx, h]  # [max_seq_len, head_dim]
                    valid_value_head = value_head[:key_valid_len]  # [valid_len_j, head_dim]
                    
                    # 计算加权值 [valid_len_i, head_dim]
                    weighted_values = torch.matmul(attn_weights, valid_value_head)
                    
                    # 将计算结果回填到输出特征中 (保持填充部分为零)
                    output_features[i, h, :query_valid_len] += weighted_values
            
        # 重塑回原始形状
        output_features = output_features.permute(0, 2, 1, 3).contiguous()  # [num_nodes, max_seq_len, num_heads, head_dim]
        output_features = output_features.reshape(num_nodes, max_seq_len, self.dim)  # [num_nodes, max_seq_len, dim]
        
        # 应用输出投影
        output = self.o_proj(output_features)
        output = self.dropout_layer(output)
        
        # 应用残差连接
        output = output + node_features
        
        # 确保填充位置保持为0 (使用精确的掩码机制)
        output = output * node_masks.unsqueeze(-1).float()
        
        return output


class GraphAttentionNetworkWithMask(nn.Module):
    """支持直接处理变长序列特征和掩码的图注意力网络"""
    
    def __init__(self, dim, num_heads, speaker_embedding_dim=None, 
                 num_speakers=None, num_layers=2, dropout=0.2):
        """
        初始化支持掩码的图注意力网络
        
        参数:
            dim: 特征维度 (输入输出维度相同)
            num_heads: 注意力头数量
            speaker_embedding_dim: 说话者嵌入维度，若为None则设为等于dim
            num_speakers: 说话者数量，None表示动态确定
            num_layers: 图注意力层数量
            dropout: Dropout概率
        """
        super(GraphAttentionNetworkWithMask, self).__init__()
        
        # 确保dim可被num_heads整除
        if dim % num_heads != 0:
            print(f"Warning: dim ({dim}) not divisible by num_heads ({num_heads}). Adjusting dim.")
            dim = (dim // num_heads) * num_heads
            if dim == 0:
                raise ValueError("dim becomes 0 after adjustment.")

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_edge_types = 4  # 固定基于描述
        
        # 若未指定speaker_embedding_dim，则设为与节点特征维度相同
        if speaker_embedding_dim is None:
            speaker_embedding_dim = dim
        
        self.speaker_embedding_dim = speaker_embedding_dim

        # 说话者嵌入层
        self.speaker_embedding = SpeakerEmbedding(speaker_embedding_dim, num_speakers)
        
        # 新增：添加特征拼接后的投影层，将拼接后的特征映射回原始维度
        self.concat_projection = nn.Linear(dim + speaker_embedding_dim, dim)
        
        # 图注意力层
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                MultiHeadAttentionWithMask(dim, num_heads, dropout)
            )

        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, node_features, node_masks, speaker_ids, edge_index, edge_type=None):
        """
        前向传播
        
        参数:
            node_features: 节点特征 [num_nodes, max_seq_len, dim]
            node_masks: 节点特征掩码 [num_nodes, max_seq_len]
            speaker_ids: 说话者ID [num_nodes]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]，如果为None则所有边被视为相同类型
            
        返回:
            处理后的节点特征 [num_nodes, max_seq_len, dim]
        """
        num_nodes, max_seq_len, _ = node_features.size()
        device = node_features.device
        
        # 获取说话者嵌入
        speaker_emb = self.speaker_embedding(speaker_ids)  # [num_nodes, speaker_embedding_dim]
        
        # 将说话者嵌入扩展到与节点特征相同的序列长度
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, max_seq_len, -1)  # [num_nodes, max_seq_len, speaker_embedding_dim]
        
        # 新增：将说话者嵌入与节点特征拼接
        combined_features = torch.cat([node_features, speaker_emb], dim=-1)  # [num_nodes, max_seq_len, dim + speaker_embedding_dim]
        
        # 新增：投影回原始维度
        x = self.concat_projection(combined_features)  # [num_nodes, max_seq_len, dim]
        
        # 应用多层图注意力
        for i, gat_layer in enumerate(self.gat_layers):
            # 应用GAT层
            x = gat_layer(x, node_masks, edge_index, edge_type)
            
            # 在层之间应用激活函数和dropout（不在最后一层之后）
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        # 确保掩码得到应用
        x = x * node_masks.unsqueeze(-1)
        
        return x


class DialogueGraphBuilder:
    """对话图构建器，负责从原始对话数据构建对话图结构（边）"""
    
    def __init__(self, similarity_threshold=0.9, context_window_size=3):
        """
        初始化对话图构建器
        
        参数:
            similarity_threshold: 相似度阈值，用于构建跨轮次关联边
            context_window_size: 上下文窗口大小，用于构建用户关联边
        """
        self.similarity_threshold = similarity_threshold
        self.context_window_size = context_window_size

    def compute_masked_similarity_matrix(self, features, masks):
        """
        使用掩码计算节点间的相似度矩阵
        
        参数:
            features: 节点特征 [num_nodes, max_seq_len, dim]
            masks: 节点掩码 [num_nodes, max_seq_len]
            
        返回:
            相似度矩阵 [num_nodes, num_nodes]
        """
        num_nodes, max_seq_len, dim = features.size()
        device = features.device
        
        if num_nodes == 0:
            return torch.zeros((0, 0), device=device)
        
        # 扩展掩码为 [num_nodes, max_seq_len, 1]
        expanded_masks = masks.unsqueeze(-1).float()
        
        # 计算每个节点的平均特征向量 (用于相似度计算)
        # [num_nodes, dim]
        mask_sum = masks.sum(dim=1, keepdim=True).clamp(min=1.0)  # 防止除零
        avg_features = (features * expanded_masks).sum(dim=1) / mask_sum
        
        # 防止特征全零导致的NaN
        zero_features = torch.all(avg_features == 0, dim=1, keepdim=True)
        if zero_features.any():
            # 为全零特征添加一个小的噪声以避免NaN
            noise = torch.randn_like(avg_features) * 1e-6
            avg_features = torch.where(zero_features, noise, avg_features)
        
        # 归一化特征，防止零向量
        feature_norms = torch.norm(avg_features, p=2, dim=1, keepdim=True)
        # 避免除以零
        feature_norms = torch.clamp(feature_norms, min=1e-8)
        normalized_features = avg_features / feature_norms
        
        # 计算余弦相似度
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # 检查并修复NaN值
        similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 限制值在 [-1, 1] 范围内
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        
        return similarity_matrix

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

        # 创建节点 (此时特征可以是 None 或 原始特征)
        if timestamps is None:
            timestamps = list(range(num_nodes))
        elif len(timestamps) != num_nodes:
             raise ValueError(f"Length of timestamps ({len(timestamps)}) must match num_nodes ({num_nodes})")

        for i in range(num_nodes):
            node = Node(node_id=i, utterance_features=None, speaker_id=speaker_ids_list[i], timestamp=timestamps[i])
            graph.add_node(node)

        # 构建时序边 (Type 1)
        graph.build_temporal_edges()

        # 构建用户关联边 (Type 2)
        graph.build_user_context_edges(self.context_window_size)

        # 构建自环边 (Type 4)
        graph.build_self_loop_edges()

        # 注意：跨轮次边 (Type 3) 在 DialogueGraphModel.forward 中构建，因为它需要特征信息

        return graph


class DialogueGraphModel(nn.Module):
    """对话图模型，集成图注意力网络和其他组件"""
    
    def __init__(self, token_embedding_dim, output_dim=None, num_heads=4,
                 speaker_embedding_dim=None, num_speakers=None, num_layers=2, dropout=0.2,
                 similarity_threshold=0.8, context_window_size=3):
        """
        初始化对话图模型
        
        参数:
            token_embedding_dim: 输入的 token 嵌入维度
            output_dim: GAT 输出特征维度, 如果为None则设置为等于 token_embedding_dim 以保持维度不变
            num_heads: 注意力头数量
            speaker_embedding_dim: 说话者嵌入维度，若为None则设为等于token_embedding_dim
            num_speakers: 说话者数量，None表示动态确定
            num_layers: 图注意力层数量
            dropout: Dropout概率
            similarity_threshold: 相似度阈值 (初始值，可以设为可学习)
            context_window_size: 上下文窗口大小
        """
        super(DialogueGraphModel, self).__init__()

        self.token_embedding_dim = token_embedding_dim
            
        # 如果未指定output_dim，默认设为等于token_embedding_dim
        if output_dim is None:
            output_dim = token_embedding_dim
            
        # 如果未指定speaker_embedding_dim，默认设为等于token_embedding_dim
        if speaker_embedding_dim is None:
            speaker_embedding_dim = token_embedding_dim
            
        # 确保token_embedding_dim可以被num_heads整除
        gat_dim = token_embedding_dim
        if gat_dim % num_heads != 0:
            print(f"Warning: token_embedding_dim ({token_embedding_dim}) not divisible by num_heads ({num_heads}). Adjusting dim.")
            gat_dim = (gat_dim // num_heads) * num_heads
            if gat_dim == 0:
                gat_dim = num_heads
                print(f"Warning: dimension was adjusted to minimum value: {gat_dim}")
                
        if output_dim % num_heads != 0:
            print(f"Warning: Adjusting output_dim from {output_dim} to be divisible by num_heads ({num_heads})")
            output_dim = (output_dim // num_heads) * num_heads
            if output_dim == 0:
                output_dim = num_heads
                print(f"Warning: output_dim was adjusted to minimum value: {output_dim}")

        # 使用支持掩码的图注意力网络
        self.gat = GraphAttentionNetworkWithMask(
            dim=gat_dim,
            num_heads=num_heads,
            speaker_embedding_dim=speaker_embedding_dim,
            num_speakers=num_speakers,
            num_layers=num_layers,
            dropout=dropout
        )

        # 对话图构建器 (用于构图逻辑)
        self.graph_builder = DialogueGraphBuilder(
            similarity_threshold=similarity_threshold,
            context_window_size=context_window_size
        )

        self.similarity_threshold_value = similarity_threshold
        
        # 存储初始化参数，方便使用
        self.hidden_dim = gat_dim
        self.output_dim = output_dim
        
        # 如果GAT使用的维度与输入不同，添加初始调整
        if gat_dim != token_embedding_dim:
            self.dim_adjust = nn.Linear(token_embedding_dim, gat_dim)
        else:
            self.dim_adjust = nn.Identity()
        
        # 输出投影层 - 将GAT输出投影回output_dim（如果需要）
        if gat_dim != output_dim:
            self.output_projection = nn.Linear(gat_dim, output_dim)
        else:
            self.output_projection = nn.Identity()

    def forward(self, utterance_features_input, speaker_ids_input, attention_masks=None, edge_index=None, edge_type=None):
        """
        前向传播，直接处理变长序列特征和掩码
        
        参数:
            utterance_features_input: 节点特征 [batch_size, max_num_segments, max_segment_len, feat_dim]
            speaker_ids_input: 说话者ID [batch_size, max_num_segments] 或 List[List[int/str]]
            attention_masks: 注意力掩码 [batch_size, max_num_segments, max_segment_len]
            edge_index: 预定义的边索引 [2, num_edges], 如果为None则自动构建图 (仅支持 batch_size=1)
            edge_type: 预定义的边类型 [num_edges], 如果为None则自动构建图 (仅支持 batch_size=1)
            
        返回:
            更新后的节点特征 [batch_size, max_num_segments, max_segment_len, output_dim]
        """
        device = next(self.parameters()).device

        # --- 验证输入 ---
        if attention_masks is None:
            raise ValueError("attention_masks must be provided")
            
        # --- 处理单批次情况 (batch_size=1) ---
        batch_size = utterance_features_input.size(0)
        if batch_size != 1:
            raise ValueError("Currently only supports batch_size=1")
            
        # 提取单批次数据
        features = utterance_features_input[0]  # [max_num_segments, max_segment_len, feat_dim]
        masks = attention_masks[0]  # [max_num_segments, max_segment_len]
        
        # 处理speaker_ids - 增强类型处理能力
        if isinstance(speaker_ids_input, torch.Tensor):
            if speaker_ids_input.dim() == 2:  # [batch_size, max_num_segments]
                speaker_ids = speaker_ids_input[0]  # [max_num_segments]
            else:
                speaker_ids = speaker_ids_input  # 假设已经是正确形状
        elif isinstance(speaker_ids_input, list):
            # 处理嵌套列表情况 (来自DataLoader的批次)
            if len(speaker_ids_input) > 0 and isinstance(speaker_ids_input[0], list):
                # 获取第一个批次的speaker_ids
                batch_speakers = speaker_ids_input[0]
                
                # 将任何非数值类型的ID转换为数值索引
                speaker_to_idx = {}
                next_idx = 0
                processed_speakers = []
                
                for speaker in batch_speakers:
                    if speaker not in speaker_to_idx:
                        speaker_to_idx[speaker] = next_idx
                        next_idx += 1
                    processed_speakers.append(speaker_to_idx[speaker])
                
                speaker_ids = torch.tensor(processed_speakers, device=device)
            else:
                # 单层列表，直接转换
                try:
                    speaker_ids = torch.tensor(speaker_ids_input, device=device)
                except (ValueError, TypeError):
                    # 处理列表中包含非数值类型的情况
                    speaker_to_idx = {}
                    next_idx = 0
                    processed_speakers = []
                    
                    for speaker in speaker_ids_input:
                        if speaker not in speaker_to_idx:
                            speaker_to_idx[speaker] = next_idx
                            next_idx += 1
                        processed_speakers.append(speaker_to_idx[speaker])
                    
                    speaker_ids = torch.tensor(processed_speakers, device=device)
        else:
            raise TypeError(f"Unsupported speaker_ids_input type: {type(speaker_ids_input)}")
        
        # 移动数据到设备
        features = features.to(device)
        masks = masks.to(device)
        speaker_ids = speaker_ids.to(device)
        
        # 如果需要，调整特征维度以满足GAT要求
        features = self.dim_adjust(features)
        
        # --- 构建图结构 ---
        num_segments = features.size(0)
        
        if edge_index is None or edge_type is None:
            # 自动构建图结构
            graph = self.graph_builder.build_graph_structure(
                num_nodes=num_segments,
                speaker_ids_list=speaker_ids.cpu().tolist(),
                timestamps=list(range(num_segments))
            )
            
            # 构建跨轮次边 (Type 3) - 基于掩码计算相似度
            if num_segments > 1:
                # 使用掩码计算节点间相似度
                similarity_matrix = self.graph_builder.compute_masked_similarity_matrix(features, masks)
                graph.build_cross_turn_edges(similarity_matrix, self.similarity_threshold_value)
                
            # 获取图张量
            edge_index, edge_type, _, _ = graph.get_tensors(device)
            
        # --- 应用图注意力网络 ---
        updated_features = self.gat(
            features,      # [max_num_segments, max_segment_len, hidden_dim]
            masks,         # [max_num_segments, max_segment_len]
            speaker_ids,   # [max_num_segments]
            edge_index,    # [2, num_edges]
            edge_type      # [num_edges]
        )
        
        # 投影输出特征到output_dim（如果需要）
        updated_features = self.output_projection(updated_features)  # [max_num_segments, max_segment_len, output_dim]
        
        # 确保填充位置的特征为0
        updated_features = updated_features * masks.unsqueeze(-1).float()
        
        # 强制掩码位置为准确的0
        if masks.numel() > 0:
            mask_inverted = ~masks.bool()
            updated_features.masked_fill_(mask_inverted.unsqueeze(-1), 0.0)

            # 1. 节点内池化 (Token Pooling)
            token_mask = masks.unsqueeze(-1).float()  # [max_num_segments, max_segment_len, 1]
            token_sum = (updated_features * token_mask).sum(dim=1)  # [max_num_segments, output_dim]
            num_valid_tokens = masks.sum(dim=1, keepdim=True).clamp(min=1e-9)  # [max_num_segments, 1]
            segment_embeddings = token_sum / num_valid_tokens  # [max_num_segments, output_dim]

            # 2. 图级别池化 (Node/Segment Pooling)
            # 创建有效 segment 的掩码 (假设没有完全填充的 segment)
            segment_mask = masks.any(dim=-1) # [max_num_segments]
            segment_mask_expanded = segment_mask.unsqueeze(-1).float() # [max_num_segments, 1]

            graph_sum = (segment_embeddings * segment_mask_expanded).sum(dim=0)  # [output_dim]
            num_valid_segments = segment_mask.sum(dim=0).clamp(min=1e-9)  # scalar
            graph_embedding = graph_sum / num_valid_segments  # [output_dim]

            # 添加批次维度
            graph_embedding = graph_embedding.unsqueeze(0) # [1, output_dim]

            # 分离梯度
            with torch.no_grad():
                result = graph_embedding.detach().clone()
        else:
            # 返回节点特征
            # 重塑为批次形式
            node_embeddings = updated_features.unsqueeze(0)  # [1, max_num_segments, max_segment_len, output_dim]
            
            # 分离梯度，确保返回的张量不带梯度
            with torch.no_grad():
                result = node_embeddings.detach().clone()
        
        return result
    
    def get_node_embeddings(self, utterance_features, speaker_ids, attention_masks):
        """
        获取对话中所有节点的嵌入表示 (non-batched, builds graph internally).
        
        参数:
            utterance_features: 话语特征 [batch_size, max_num_segments, max_segment_len, feat_dim]
            speaker_ids: 说话者ID [batch_size, max_num_segments] 或 List[int]
            attention_masks: 注意力掩码 [batch_size, max_num_segments, max_segment_len]
            
        返回:
            每个节点的更新特征 [max_num_segments, max_segment_len, output_dim] (移除批次维度)
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            # 使用forward方法处理输入, 获取节点嵌入
            node_features = self.forward(
                utterance_features,
                speaker_ids,
                attention_masks=attention_masks
            )
            
            # 删除批次维度
            node_features = node_features.squeeze(0)  # [max_num_segments, max_segment_len, output_dim]
            
        # 确保没有梯度
        return node_features.detach()

    def get_graph_embedding(self, utterance_features, speaker_ids, attention_masks):
        """
        获取整个对话图的嵌入表示 (non-batched, builds graph internally).
        
        参数:
            utterance_features: 话语特征 [batch_size, max_num_segments, max_segment_len, feat_dim]
            speaker_ids: 说话者ID [batch_size, max_num_segments] 或 List[int]
            attention_masks: 注意力掩码 [batch_size, max_num_segments, max_segment_len]
            
        返回:
            图的嵌入表示 [output_dim] (移除批次维度)
        """
        self.eval() # 设置为评估模式
        with torch.no_grad():
            # 使用forward方法处理输入
            # forward 方法现在默认返回图嵌入
            graph_embedding = self.forward(
                utterance_features,
                speaker_ids,
                attention_masks=attention_masks
            )
            
            # 删除批次维度
            graph_embedding = graph_embedding.squeeze(0) # [output_dim]
            
        # 确保没有梯度
        return graph_embedding.detach()