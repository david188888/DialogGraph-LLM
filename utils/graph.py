import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataclasses
from typing import Optional, Union, List, Dict, Any


class DialogueGraphConfig:
    """Configuration class for the dialogue graph model, containing all parameters for initializing DialogueGraphModel"""
    
    # Model structure parameters
    token_embedding_dim: int = 768  # Input token embedding dimension
    output_dim: Optional[int] = None  # Output feature dimension, if None equals token_embedding_dim
    num_heads: int = 4  # Number of attention heads
    speaker_embedding_dim: Optional[int] = 16  # Speaker embedding dimension
    num_speakers: Optional[int] = None  # Number of speakers, if None determined dynamically
    num_layers: int = 2  # Number of GAT layers
    
    # Graph structure parameters
    similarity_threshold: float = 0.8  # Similarity threshold for building cross-turn association edges
    context_window_size: int = 4  # Context window size for user association edges
    
    # Training parameters
    dropout: float = 0.1  # Dropout probability
    
    def asdict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for passing to model initialization function"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DialogueGraphConfig":
        """Create configuration instance from dictionary"""
        return cls(**config_dict)


class Node:
    """Node class, representing an utterance in a dialogue"""
    
    def __init__(self, node_id, utterance_features, speaker_id, timestamp=None, attention_mask=None):
        """
        Initialize node
        
        Parameters:
            node_id: Unique node identifier
            utterance_features: Multimodal fusion features of the utterance (can be a tensor of shape (token_num, dim))
            speaker_id: Speaker ID
            timestamp: Timestamp, used to maintain temporal information
            attention_mask: Attention mask, marking which positions are valid (token_num,)
        """
        self.id = node_id
        self.utterance_features = utterance_features  # Original features, possibly of shape (token_num, dim)
        self.speaker_id = speaker_id
        self.timestamp = timestamp
        self.embedding = None  # Final node embedding output by GAT
        self.attention_mask = attention_mask  # Attention mask
    
    def set_embedding(self, embedding):
        """Set the final embedding representation of the node"""
        self.embedding = embedding
        
    def get_embedding(self):
        """Get the embedding representation of the node"""
        return self.embedding


class Edge:
    """Edge class, representing relationships between nodes"""
    
    # Edge type IDs
    TEMPORAL_EDGE = 1  # Historical temporal edge
    USER_CONTEXT_EDGE = 2  # User association edge
    CROSS_TURN_EDGE = 3  # Cross-turn association edge
    SELF_LOOP_EDGE = 4  # Self-loop edge
    
    def __init__(self, source_id, target_id, edge_type, weight=None):
        """
        Initialize edge
        
        Parameters:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type (1-4)
            weight: Edge weight, defaults to None, later calculated through attention mechanism
        """
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
    
    def set_weight(self, weight):
        """Set edge weight"""
        self.weight = weight
        
    def get_type_name(self):
        """Get edge type name"""
        if self.edge_type == self.TEMPORAL_EDGE:
            return "Temporal Edge"
        elif self.edge_type == self.USER_CONTEXT_EDGE:
            return "User Context Edge"
        elif self.edge_type == self.CROSS_TURN_EDGE:
            return "Cross-Turn Edge"
        elif self.edge_type == self.SELF_LOOP_EDGE:
            return "Self-Loop Edge"
        else:
            return "Unknown Edge Type"


class SpeakerEmbedding(nn.Module):
    """Speaker embedding module, builds learnable embedding vectors for each speaker"""
    
    def __init__(self, embedding_dim, num_speakers=None):
        """
        Initialize speaker embedding module
        
        Parameters:
            embedding_dim: Embedding dimension, will be the same as node feature dimension
            num_speakers: Optional, number of speakers; if not specified, handled dynamically
        """
        super(SpeakerEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        
        # Dynamic speaker mapping dictionary
        self.speaker_to_idx = {}
        self.next_idx = 0
        
        # Use embedding layer instead of linear layer to handle variable-length speaker IDs
        if num_speakers is not None:
            self.embedding = nn.Embedding(num_speakers, embedding_dim)
        else:
            # Initial capacity of 2, dynamically expanded later
            self.embedding = nn.Embedding(2, embedding_dim)
    
    def _expand_embedding_if_needed(self, max_id):
        """Expand embedding layer capacity as needed"""
        current_size = self.embedding.num_embeddings
        if max_id >= current_size:
            # Create new embedding layer with larger capacity
            new_size = max(max_id + 1, current_size * 2)  # At least double the size
            new_embedding = nn.Embedding(new_size, self.embedding_dim)
            
            # Copy old weights
            with torch.no_grad():
                new_embedding.weight[:current_size] = self.embedding.weight
            
            # Replace old embedding layer
            self.embedding = new_embedding.to(self.embedding.weight.device)
            
    def forward(self, speaker_ids):
        """
        Compute speaker embeddings
        
        Parameters:
            speaker_ids: List or tensor of speaker IDs [N]
            
        Returns:
            Speaker embedding vectors [N, embedding_dim]
        """
        # Ensure input is on the correct device
        device = self.embedding.weight.device
        
        # Handle list input
        if isinstance(speaker_ids, list):
            # Dynamically map speaker IDs
            if self.num_speakers is None:
                indices = []
                for speaker_id in speaker_ids:
                    if speaker_id not in self.speaker_to_idx:
                        self.speaker_to_idx[speaker_id] = self.next_idx
                        self.next_idx += 1
                    indices.append(self.speaker_to_idx[speaker_id])
                
                # Check if embedding layer needs expansion
                max_idx = max(indices) if indices else -1
                if max_idx >= self.embedding.num_embeddings:
                    self._expand_embedding_if_needed(max_idx)
                
                # Convert to tensor
                speaker_indices = torch.tensor(indices, dtype=torch.long, device=device)
            else:
                # Fixed number of speakers, use ID directly as index
                speaker_indices = torch.tensor(speaker_ids, dtype=torch.long, device=device)
                
        # Handle tensor input
        elif isinstance(speaker_ids, torch.Tensor):
            speaker_ids = speaker_ids.to(device)
            
            # Dynamic mapping
            if self.num_speakers is None:
                unique_ids = speaker_ids.unique().cpu().tolist()
                for speaker_id in unique_ids:
                    if speaker_id not in self.speaker_to_idx:
                        self.speaker_to_idx[speaker_id] = self.next_idx
                        self.next_idx += 1
                
                # Map IDs to indices
                indices = [self.speaker_to_idx[sid.item()] for sid in speaker_ids]
                max_idx = max(indices) if indices else -1
                if max_idx >= self.embedding.num_embeddings:
                    self._expand_embedding_if_needed(max_idx)
                
                speaker_indices = torch.tensor(indices, dtype=torch.long, device=device)
            else:
                # Fixed number of speakers, use directly
                speaker_indices = speaker_ids.long()
        else:
            raise ValueError(f"Unsupported speaker_ids type: {type(speaker_ids)}")
        
        # Get embedding vectors
        return self.embedding(speaker_indices)


class DialogueGraph:
    """Dialogue graph, representing the complete graph structure of a dialogue"""
    
    def __init__(self):
        """Initialize dialogue graph"""
        self.nodes = {}  # Node dictionary, key is node ID, value is Node object
        self.edges = []  # Edge list
        self.edge_set = set()  # Set for quickly checking if an edge exists
        # Note: graph tensors (node_features, edge_index, edge_type) are now built dynamically in the model's forward pass
        
    def add_node(self, node):
        """Add node to graph"""
        self.nodes[node.id] = node
        
    def add_edge(self, edge):
        """Add edge to graph, skip if edge already exists"""
        # Create unique identifier for edge (source node ID, target node ID, edge type)
        edge_key = (edge.source_id, edge.target_id, edge.edge_type)
        
        # Check if edge already exists
        if edge_key not in self.edge_set:
            self.edges.append(edge)
            self.edge_set.add(edge_key)
        
    def build_temporal_edges(self):
        """
        Build historical temporal edges, connecting node i to node i+1
        All edges are directed, pointing from historical nodes to future nodes
        Using updated add_edge method to avoid creating duplicate edges
        """
        nodes_list = sorted(self.nodes.values(), key=lambda x: x.timestamp if x.timestamp is not None else x.id)
        for i in range(len(nodes_list) - 1):
            # i -> i+1, from past to future
            edge = Edge(nodes_list[i].id, nodes_list[i+1].id, Edge.TEMPORAL_EDGE)
            self.add_edge(edge)
            
    def build_user_context_edges(self, window_size=4):
        """
        Build user association edges (Type 2)
        For each user utterance node, we build a context window of size k for past nodes,
        and build edges from historical user utterance nodes in this window to the current user utterance node.
        In simpler terms: find the k most recent nodes with the same speaker in history, and build edges from them to the current node.
        
        Parameters:
            window_size: Context window size
        """
        # Sort nodes by timestamp
        nodes_list = sorted(self.nodes.values(), key=lambda x: x.timestamp if x.timestamp is not None else x.id)
        
        # Traverse all nodes
        for i in range(1, len(nodes_list)):  # Start from the second node, as the first has no historical nodes
            current_node = nodes_list[i]
            current_speaker = current_node.speaker_id
            
            # Record the number of same speaker nodes found
            found_same_speaker = 0
            
            # Look for window_size nodes with the same speaker going backwards
            for j in range(i-1, -1, -1):  # Traverse from i-1 backwards
                prev_node = nodes_list[j]
                
                # If a node with the same speaker is found
                if prev_node.speaker_id == current_speaker:
                    # Build an edge from the historical node to the current node
                    edge = Edge(prev_node.id, current_node.id, Edge.USER_CONTEXT_EDGE)
                    self.add_edge(edge)
                    
                    found_same_speaker += 1
                    
                    # If window_size nodes with the same speaker are found, stop searching
                    if found_same_speaker >= window_size:
                        break

    def build_cross_turn_edges(self, similarity_matrix, threshold):
        """
        Build cross-turn association edges (Type 3)
        Original definition: User utterance node i connects to previous service or user utterance node j (j < i), if similarity > threshold, then j -> i.
        
        Parameters:
            similarity_matrix: Node similarity matrix (calculated using aggregated features) [N, N]
            threshold: Similarity threshold
        """
        nodes_list = sorted(self.nodes.values(), key=lambda x: x.timestamp if x.timestamp is not None else x.id)
        node_ids = [node.id for node in nodes_list]
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        num_nodes = len(nodes_list)
        for i in range(num_nodes): # Current node i
            # TODO: Add logic to check if nodes_list[i] is a user node if needed
            
            for j in range(i): # Previous node j
                # Get node indices in matrix
                idx_i = node_id_to_idx[nodes_list[i].id]
                idx_j = node_id_to_idx[nodes_list[j].id]
                
                # If similarity is greater than threshold, build edge j -> i
                if similarity_matrix[idx_j, idx_i] > threshold:
                    # Source node j can be user or service
                    edge = Edge(nodes_list[j].id, nodes_list[i].id, Edge.CROSS_TURN_EDGE)
                    self.add_edge(edge)
                    # # # (Optional) Add reverse edge i -> j
                    # edge_rev = Edge(nodes_list[i].id, nodes_list[j].id, Edge.CROSS_TURN_EDGE)
                    # self.add_edge(edge_rev)

    def build_self_loop_edges(self):
        """
        Build self-loop edges (Type 4)
        """
        for node_id in self.nodes:
            edge = Edge(node_id, node_id, Edge.SELF_LOOP_EDGE)
            self.add_edge(edge)
            
    def get_tensors(self, device):
        """
        Extract edge information of graph structure as tensors, feature aggregation is done in model's forward pass.
        
        Returns:
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
            node_ids_map: Mapping from original node IDs to their indices in the sorted list
            sorted_node_ids: Sorted list of node IDs
        """
        if not self.nodes:
            # Handle empty graph case
            return (torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.long, device=device),
                    {}, [])

        # Sort nodes by timestamp (or ID) for consistency
        sorted_nodes = sorted(self.nodes.values(), key=lambda node: node.timestamp if node.timestamp is not None else node.id)
        sorted_node_ids = [node.id for node in sorted_nodes]
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}

        edge_index = []
        edge_type = []
        for edge in self.edges:
            # Get node indices in sorted list
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
    """Multi-head attention mechanism that directly supports variable-length sequence features and masks"""
    
    def __init__(self, dim, num_heads, dropout=0.2, num_edge_types=4):
        """
        Initialize masked multi-head attention module
        
        Parameters:
            dim: Feature dimension (input and output dimensions are the same)
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_edge_types: Number of edge types, default is 4
        """
        super(MultiHeadAttentionWithMask, self).__init__()
        
        # Ensure dim is divisible by num_heads
        if dim % num_heads != 0:
            print(f"Warning: dim ({dim}) not divisible by num_heads ({num_heads}). Adjusting dim.")
            dim = (dim // num_heads) * num_heads
            if dim == 0:
                raise ValueError("dim becomes 0 after adjustment.")
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        
        # Add learnable weight parameters for each edge type
        # Initialize to all 1s, indicating all edge types have the same importance initially
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(dim)
        
        # Projection layers - keep input and output dimensions the same
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, node_features, node_masks, edge_index, edge_type=None):
        """
        Forward pass, directly handling variable-length sequence features and masks
        
        Parameters:
            node_features: Original node feature list [num_nodes, max_seq_len, dim]
            node_masks: Node feature masks [num_nodes, max_seq_len]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges], if None then all edges are treated as the same type
            
        Returns:
            Updated node features [num_nodes, max_seq_len, dim]
        """
        num_nodes, max_seq_len, hidden_dim = node_features.size()
        device = node_features.device
        
        if num_nodes == 0 or edge_index.numel() == 0:
            # Handle empty graph or no edge case
            return torch.zeros_like(node_features)
        
        # Build graph adjacency matrix (with edge type weights)
        adj_matrix = torch.full((num_nodes, num_nodes), float('-inf'), device=device)
        
        src_nodes, dst_nodes = edge_index
        
        if edge_type is not None:
            # Use edge type weights
            for i in range(edge_index.shape[1]):
                src, dst = src_nodes[i], dst_nodes[i]
                # Get edge type (starting from 1), subtract 1 as index (starting from 0)
                e_type_idx = edge_type[i].item() - 1
                # Use softplus to ensure weight is positive, and ensure stability
                weight = F.softplus(self.edge_type_weights[e_type_idx]).clamp(min=1e-6, max=1e6)
                # Apply edge weight as value in adjacency matrix
                adj_matrix[src, dst] = weight
        else:
            # If edge type not provided, all edges use the same weight
            adj_matrix[src_nodes, dst_nodes] = 1.0
        
        # Apply layer normalization
        normed_features = self.layer_norm(node_features)
        
        # Calculate query, key, and value
        q = self.q_proj(normed_features)  # [num_nodes, max_seq_len, dim]
        k = self.k_proj(normed_features)  # [num_nodes, max_seq_len, dim]
        v = self.v_proj(normed_features)  # [num_nodes, max_seq_len, dim]
        
        # Reshape to multi-head form - use reshape instead of view
        head_dim = self.dim // self.num_heads
        q = q.reshape(num_nodes, max_seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [num_nodes, num_heads, max_seq_len, head_dim]
        k = k.reshape(num_nodes, max_seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [num_nodes, num_heads, max_seq_len, head_dim]
        v = v.reshape(num_nodes, max_seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [num_nodes, num_heads, max_seq_len, head_dim]
        
        # Get valid token count for each node (for subsequent calculations)
        valid_tokens_per_node = node_masks.sum(dim=1).long()  # [num_nodes]
        
        # Create output feature tensor, initialized to zero
        # Shape: [num_nodes, num_heads, max_seq_len, head_dim]
        output_features = torch.zeros_like(q)
        
        # Calculate attention for each node individually
        for i in range(num_nodes):
            # Skip padding nodes (if any)
            if valid_tokens_per_node[i] == 0:
                continue
                
            # Get current node's query representation and mask
            # Only keep valid tokens (determined by mask)
            query_node_mask = node_masks[i]  # [max_seq_len]
            query_valid_len = valid_tokens_per_node[i]
            
            # For this node, collect information from all connected nodes
            attending_nodes = []
            edge_weights = []
            
            # Find all nodes connected to node i (edge i->j exists)
            for j in range(num_nodes):
                if adj_matrix[i, j] != float('-inf'):
                    attending_nodes.append(j)
                    edge_weights.append(adj_matrix[i, j])
            
            if not attending_nodes:
                continue  # If no connected nodes, skip
                
            # Convert lists to tensors for easier operations
            attending_nodes = torch.tensor(attending_nodes, device=device)
            edge_weights = torch.tensor(edge_weights, device=device)
            
            # Get key and value vectors for all connected nodes
            # Shape: [num_attending, num_heads, max_seq_len, head_dim]
            keys_attending = k[attending_nodes]
            values_attending = v[attending_nodes]
            
            # Get masks for connected nodes
            # Shape: [num_attending, max_seq_len]
            masks_attending = node_masks[attending_nodes]
            
            # Calculate attention scores from node i to all connected nodes
            # Calculate separately for each head
            for h in range(self.num_heads):
                # Current node's query vector for current head [max_seq_len, head_dim]
                query_head = q[i, h]  
                
                # Only calculate attention for valid tokens
                valid_query_head = query_head[:query_valid_len]  # [valid_len, head_dim]
                
                # Store attention values for all connected nodes
                num_attending = len(attending_nodes)
                
                # For each connected node j
                for idx, j in enumerate(attending_nodes):
                    # Get valid token count for node j
                    key_node_mask = masks_attending[idx]  # [max_seq_len]
                    key_valid_len = key_node_mask.sum().long()
                    
                    if key_valid_len == 0:
                        continue  # Skip nodes with no valid tokens
                    
                    # Get key vector for current connected node - only valid part
                    key_head = keys_attending[idx, h]  # [max_seq_len, head_dim]
                    valid_key_head = key_head[:key_valid_len]  # [valid_len_j, head_dim]
                    
                    # Calculate attention scores for this node pair [valid_len_i, valid_len_j]
                    scores = torch.matmul(valid_query_head, valid_key_head.transpose(0, 1)) / (head_dim ** 0.5)
                    
                    # Add edge weight (same bias for all scores)
                    scores = scores + edge_weights[idx]
                    
                    # Apply softmax to scores for each query position - key modification!
                    # This way each valid query token only attends to valid key tokens
                    attn_weights = F.softmax(scores, dim=1)  # [valid_len_i, valid_len_j]
                    
                    # Handle noise
                    attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Get value vector for current connected node - only valid part
                    value_head = values_attending[idx, h]  # [max_seq_len, head_dim]
                    valid_value_head = value_head[:key_valid_len]  # [valid_len_j, head_dim]
                    
                    # Calculate weighted values [valid_len_i, head_dim]
                    weighted_values = torch.matmul(attn_weights, valid_value_head)
                    
                    # Fill calculation results back to output features (keep padding part as zero)
                    output_features[i, h, :query_valid_len] += weighted_values
            
        # Reshape back to original shape
        output_features = output_features.permute(0, 2, 1, 3).contiguous()  # [num_nodes, max_seq_len, num_heads, head_dim]
        output_features = output_features.reshape(num_nodes, max_seq_len, self.dim)  # [num_nodes, max_seq_len, dim]
        
        # Apply output projection
        output = self.o_proj(output_features)
        output = self.dropout_layer(output)
        
        # Apply residual connection
        output = output + node_features
        
        # Ensure padding positions remain 0 (using precise mask mechanism)
        output = output * node_masks.unsqueeze(-1).to(dtype=node_features.dtype)
        
        return output


class GraphAttentionNetworkWithMask(nn.Module):
    """Graph attention network that directly supports variable-length sequence features and masks"""
    
    def __init__(self, dim, num_heads, speaker_embedding_dim=None, 
                 num_speakers=None, num_layers=2, dropout=0.2):
        """
        Initialize masked graph attention network
        
        Parameters:
            dim: Feature dimension (input and output dimensions are the same)
            num_heads: Number of attention heads
            speaker_embedding_dim: Speaker embedding dimension, if None set to equal dim
            num_speakers: Number of speakers, None indicates dynamic determination
            num_layers: Number of graph attention layers
            dropout: Dropout probability
        """
        super(GraphAttentionNetworkWithMask, self).__init__()
        
        # Ensure dim is divisible by num_heads
        if dim % num_heads != 0:
            print(f"Warning: dim ({dim}) not divisible by num_heads ({num_heads}). Adjusting dim.")
            dim = (dim // num_heads) * num_heads
            if dim == 0:
                raise ValueError("dim becomes 0 after adjustment.")

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_edge_types = 4  # Fixed based on description
        
        # If speaker_embedding_dim not specified, set to same as node feature dimension
        if speaker_embedding_dim is None:
            speaker_embedding_dim = dim
        
        self.speaker_embedding_dim = speaker_embedding_dim

        # Speaker embedding layer
        self.speaker_embedding = SpeakerEmbedding(speaker_embedding_dim, num_speakers)
        
        # New: Add projection layer after feature concatenation, mapping concatenated features back to original dimension
        self.concat_projection = nn.Linear(dim + speaker_embedding_dim, dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                MultiHeadAttentionWithMask(dim, num_heads, dropout)
            )

        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, node_features, node_masks, speaker_ids, edge_index, edge_type=None):
        """
        Forward pass
        
        Parameters:
            node_features: Node features [num_nodes, max_seq_len, dim]
            node_masks: Node feature masks [num_nodes, max_seq_len]
            speaker_ids: Speaker IDs [num_nodes]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges], if None then all edges are treated as the same type
            
        Returns:
            Processed node features [num_nodes, max_seq_len, dim]
        """
        num_nodes, max_seq_len, _ = node_features.size()
        device = node_features.device
        
        # Get speaker embeddings
        speaker_emb = self.speaker_embedding(speaker_ids)  # [num_nodes, speaker_embedding_dim]
        
        # Expand speaker embeddings to same sequence length as node features
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, max_seq_len, -1)  # [num_nodes, max_seq_len, speaker_embedding_dim]
        
        # New: Concatenate speaker embeddings with node features
        combined_features = torch.cat([node_features, speaker_emb], dim=-1)  # [num_nodes, max_seq_len, dim + speaker_embedding_dim]
        
        # New: Project back to original dimension
        x = self.concat_projection(combined_features)  # [num_nodes, max_seq_len, dim]
        
        # Apply multiple graph attention layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply GAT layer
            x = gat_layer(x, node_masks, edge_index, edge_type)
            
            # Apply activation function and dropout between layers (not after the last layer)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        # Ensure mask is applied
        x = x * node_masks.unsqueeze(-1)
        
        return x


class DialogueGraphBuilder:
    """Dialogue graph builder, responsible for constructing dialogue graph structure (edges) from raw dialogue data"""
    
    def __init__(self, similarity_threshold=0.9, context_window_size=3):
        """
        Initialize dialogue graph builder
        
        Parameters:
            similarity_threshold: Similarity threshold for building cross-turn association edges
            context_window_size: Context window size for building user association edges
        """
        self.similarity_threshold = similarity_threshold
        self.context_window_size = context_window_size

    def compute_masked_similarity_matrix(self, features, masks):
        """
        Calculate node similarity matrix using masks
        
        Parameters:
            features: Node features [num_nodes, max_seq_len, dim]
            masks: Node masks [num_nodes, max_seq_len]
            
        Returns:
            Similarity matrix [num_nodes, num_nodes]
        """
        num_nodes, max_seq_len, dim = features.size()
        device = features.device
        
        if num_nodes == 0:
            return torch.zeros((0, 0), device=device)
        
        # Expand masks to [num_nodes, max_seq_len, 1]
        expanded_masks = masks.unsqueeze(-1).to(dtype=features.dtype)
        
        # Calculate average feature vector for each node (for similarity calculation)
        # [num_nodes, dim]
        mask_sum = masks.sum(dim=1, keepdim=True).clamp(min=1.0)  # Prevent division by zero
        avg_features = (features * expanded_masks).sum(dim=1) / mask_sum
        
        # Prevent NaN caused by all-zero features
        zero_features = torch.all(avg_features == 0, dim=1, keepdim=True)
        if zero_features.any():
            # Add small noise to all-zero features to avoid NaN
            noise = torch.randn_like(avg_features) * 1e-6
            avg_features = torch.where(zero_features, noise, avg_features)
        
        # Normalize features, preventing zero vectors
        feature_norms = torch.norm(avg_features, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        feature_norms = torch.clamp(feature_norms, min=1e-8)
        normalized_features = avg_features / feature_norms
        
        # Calculate cosine similarity
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # Check and fix NaN values
        similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Limit values to [-1, 1] range
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        
        return similarity_matrix

    def build_graph_structure(self, num_nodes, speaker_ids_list, timestamps=None):
        """
        Build graph structure (nodes and edges except cross-turn edges)
        
        Parameters:
            num_nodes: Number of nodes
            speaker_ids_list: List of speaker IDs [N]
            timestamps: List of timestamps [N] (if None, use indices)
            
        Returns:
            Built dialogue graph object (only containing nodes, temporal edges, user association edges, self-loop edges)
        """
        graph = DialogueGraph()

        # Create nodes (features can be None or original features at this point)
        if timestamps is None:
            timestamps = list(range(num_nodes))
        elif len(timestamps) != num_nodes:
             raise ValueError(f"Length of timestamps ({len(timestamps)}) must match num_nodes ({num_nodes})")

        for i in range(num_nodes):
            node = Node(node_id=i, utterance_features=None, speaker_id=speaker_ids_list[i], timestamp=timestamps[i])
            graph.add_node(node)

        # Build temporal edges (Type 1)
        graph.build_temporal_edges()

        # Build user association edges (Type 2)
        graph.build_user_context_edges(self.context_window_size)

        # Build self-loop edges (Type 4)
        graph.build_self_loop_edges()

        # Note: Cross-turn edges (Type 3) are built in DialogueGraphModel.forward because they require feature information

        return graph


class DialogueGraphModel(nn.Module):
    """Dialogue graph model, integrating graph attention network and other components"""
    
    def __init__(self, token_embedding_dim=None, output_dim=None, num_heads=4,
                 speaker_embedding_dim=None, num_speakers=None, num_layers=2, dropout=0.2,
                 similarity_threshold=0.8, context_window_size=3, config=None,
                 dtype=None):
        """
        Initialize dialogue graph model
        
        Parameters:
            token_embedding_dim: Input token embedding dimension
            output_dim: GAT output feature dimension, if None set to equal token_embedding_dim to maintain dimension
            num_heads: Number of attention heads
            speaker_embedding_dim: Speaker embedding dimension, if None set to equal token_embedding_dim
            num_speakers: Number of speakers, None indicates dynamic determination
            num_layers: Number of graph attention layers
            dropout: Dropout probability
            similarity_threshold: Similarity threshold (initial value, can be set to learnable)
            context_window_size: Context window size
            config: DialogueGraphConfig instance, if provided use its parameter values
            dtype: Specify computation precision, such as torch.bfloat16 or torch.float32
        """
        super(DialogueGraphModel, self).__init__()

        # Set computation precision
        self.dtype = dtype

        # If config is provided, use parameters from config
        if config is not None:
            if isinstance(config, dict):
                config = DialogueGraphConfig.from_dict(config)
            elif not isinstance(config, DialogueGraphConfig):
                raise ValueError(f"config must be a DialogueGraphConfig instance or dictionary, not {type(config)}")
            
            token_embedding_dim = config.token_embedding_dim
            output_dim = config.output_dim
            num_heads = config.num_heads
            speaker_embedding_dim = config.speaker_embedding_dim
            num_speakers = config.num_speakers
            num_layers = config.num_layers
            dropout = config.dropout
            similarity_threshold = config.similarity_threshold
            context_window_size = config.context_window_size
        
        # If token_embedding_dim not specified, use default value
        if token_embedding_dim is None:
            token_embedding_dim = 768  # Default 768 dimensions, compatible with common pretrained models
            
        self.token_embedding_dim = token_embedding_dim
            
        # If output_dim not specified, default to token_embedding_dim
        if output_dim is None:
            output_dim = token_embedding_dim
            
        # If speaker_embedding_dim not specified, default to token_embedding_dim
        if speaker_embedding_dim is None:
            speaker_embedding_dim = token_embedding_dim
            
        # Ensure token_embedding_dim is divisible by num_heads
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

        # Use graph attention network with mask support
        self.gat = GraphAttentionNetworkWithMask(
            dim=gat_dim,
            num_heads=num_heads,
            speaker_embedding_dim=speaker_embedding_dim,
            num_speakers=num_speakers,
            num_layers=num_layers,
            dropout=dropout
        )

        # Dialogue graph builder (for graph construction logic)
        self.graph_builder = DialogueGraphBuilder(
            similarity_threshold=similarity_threshold,
            context_window_size=context_window_size
        )

        self.similarity_threshold_value = similarity_threshold
        
        # Store initialization parameters for convenience
        self.hidden_dim = gat_dim
        self.output_dim = output_dim
        
        # If GAT dimension differs from input, add initial adjustment
        if gat_dim != token_embedding_dim:
            self.dim_adjust = nn.Linear(token_embedding_dim, gat_dim)
        else:
            self.dim_adjust = nn.Identity()
        
        # Output projection layer - project GAT output back to output_dim (if needed)
        if gat_dim != output_dim:
            self.output_projection = nn.Linear(gat_dim, output_dim)
        else:
            self.output_projection = nn.Identity()
            
    @classmethod
    def from_config(cls, config):
        """Create model instance from DialogueGraphConfig"""
        if isinstance(config, dict):
            config = DialogueGraphConfig.from_dict(config)
        return cls(config=config)

    def _cast_to_dtype(self, tensor):
        """Convert tensor to specified data type"""
        if self.dtype is not None and tensor.dtype != torch.bool and tensor.dtype != torch.long:
            return tensor.to(dtype=self.dtype)
        return tensor

    def forward(self, utterance_features_input, speaker_ids_input, attention_masks=None, edge_index=None, edge_type=None):
        """
        Forward pass, directly handling variable-length sequence features and masks
        
        Parameters:
            utterance_features_input: Node features [batch_size, max_num_segments, max_segment_len, feat_dim]
            speaker_ids_input: Speaker IDs [batch_size, max_num_segments] or List[List[int/str]]
            attention_masks: Attention masks [batch_size, max_num_segments, max_segment_len]
            edge_index: Predefined edge indices [2, num_edges], if None graph is automatically built (only supports batch_size=1)
            edge_type: Predefined edge types [num_edges], if None graph is automatically built (only supports batch_size=1)
            
        Returns:
            Updated node features [batch_size, max_num_segments, max_segment_len, output_dim]
        """
        device = next(self.parameters()).device

        # --- Validate input ---
        if attention_masks is None:
            raise ValueError("attention_masks must be provided")
            
        # --- Handle single batch case (batch_size=1) ---
        batch_size = utterance_features_input.size(0)
        if batch_size != 1:
            raise ValueError("Currently only supports batch_size=1")
            
        # Extract single batch data
        features = utterance_features_input[0]  # [max_num_segments, max_segment_len, feat_dim]
        masks = attention_masks[0]  # [max_num_segments, max_segment_len]
        
        # Process speaker_ids - enhance type handling capability
        if isinstance(speaker_ids_input, torch.Tensor):
            if speaker_ids_input.dim() == 2:  # [batch_size, max_num_segments]
                speaker_ids = speaker_ids_input[0]  # [max_num_segments]
            else:
                speaker_ids = speaker_ids_input  # Assume already correct shape
        elif isinstance(speaker_ids_input, list):
            # Handle nested list case (batch from DataLoader)
            if len(speaker_ids_input) > 0 and isinstance(speaker_ids_input[0], list):
                # Get speaker_ids from first batch
                batch_speakers = speaker_ids_input[0]
                
                # Convert any non-numeric type IDs to numeric indices
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
                # Single layer list, convert directly
                try:
                    speaker_ids = torch.tensor(speaker_ids_input, device=device)
                except (ValueError, TypeError):
                    # Handle case where list contains non-numeric types
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
        
        # Move data to device and convert precision
        features = self._cast_to_dtype(features.to(device))
        masks = masks.to(device)  # Masks remain boolean or integer
        speaker_ids = speaker_ids.to(device)
        
        # If needed, adjust feature dimension to meet GAT requirements
        features = self.dim_adjust(features)
        
        # --- Build graph structure ---
        num_segments = features.size(0)
        
        if edge_index is None or edge_type is None:
            # Automatically build graph structure
            graph = self.graph_builder.build_graph_structure(
                num_nodes=num_segments,
                speaker_ids_list=speaker_ids.cpu().tolist(),
                timestamps=list(range(num_segments))
            )
            
            # Build cross-turn edges (Type 3) - based on mask-computed similarity
            if num_segments > 1:
                # Calculate node similarity using masks
                similarity_matrix = self.graph_builder.compute_masked_similarity_matrix(features, masks)
                graph.build_cross_turn_edges(similarity_matrix, self.similarity_threshold_value)
                
            # Get graph tensors
            edge_index, edge_type, _, _ = graph.get_tensors(device)
            
        # --- Apply graph attention network ---
        updated_features = self.gat(
            features,      # [max_num_segments, max_segment_len, hidden_dim]
            masks,         # [max_num_segments, max_segment_len]
            speaker_ids,   # [max_num_segments]
            edge_index,    # [2, num_edges]
            edge_type      # [num_edges]
        )
        
        # Project output features to output_dim (if needed)
        updated_features = self.output_projection(updated_features)  # [max_num_segments, max_segment_len, output_dim]
        
        # Ensure padding positions have features set to 0
        updated_features = updated_features * masks.unsqueeze(-1).float()
        
        # Force masked positions to be exactly 0
        if masks.numel() > 0:
            mask_inverted = ~masks.bool()
            updated_features.masked_fill_(mask_inverted.unsqueeze(-1), 0.0)

            # 1. Intra-node pooling (Token Pooling)
            token_mask = masks.unsqueeze(-1).float()  # [max_num_segments, max_segment_len, 1]
            token_sum = (updated_features * token_mask).sum(dim=1)  # [max_num_segments, output_dim]
            num_valid_tokens = masks.sum(dim=1, keepdim=True).clamp(min=1e-9)  # [max_num_segments, 1]
            segment_embeddings = token_sum / num_valid_tokens  # [max_num_segments, output_dim]

            # 2. Graph-level pooling (Node/Segment Pooling)
            # Create mask for valid segments (assuming no completely padded segments)
            segment_mask = masks.any(dim=-1) # [max_num_segments]
            segment_mask_expanded = segment_mask.unsqueeze(-1).float() # [max_num_segments, 1]

            graph_sum = (segment_embeddings * segment_mask_expanded).sum(dim=0)  # [output_dim]
            num_valid_segments = segment_mask.sum(dim=0).clamp(min=1e-9)  # scalar
            graph_embedding = graph_sum / num_valid_segments  # [output_dim]

            # Ensure output has correct dtype
            graph_embedding = graph_embedding.unsqueeze(0).to(dtype=updated_features.dtype) # [1, output_dim]

            # Detach gradient
            with torch.no_grad():
                result = graph_embedding.detach().clone()
        else:
            # Return node features
            # Reshape to batch form
            node_embeddings = updated_features.unsqueeze(0)  # [1, max_num_segments, max_segment_len, output_dim]
            
            # Detach gradient, ensure returned tensor has no gradient
            with torch.no_grad():
                result = node_embeddings.detach().clone()
        
        return result
    
    def get_node_embeddings(self, utterance_features, speaker_ids, attention_masks):
        """
        Get embedding representations for all nodes in the dialogue (non-batched, builds graph internally).
        
        Parameters:
            utterance_features: Utterance features [batch_size, max_num_segments, max_segment_len, feat_dim]
            speaker_ids: Speaker IDs [batch_size, max_num_segments] or List[int]
            attention_masks: Attention masks [batch_size, max_num_segments, max_segment_len]
            
        Returns:
            Updated features for each node [max_num_segments, max_segment_len, output_dim] (batch dimension removed)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Use forward method to process input, get node embeddings
            node_features = self.forward(
                utterance_features,
                speaker_ids,
                attention_masks=attention_masks
            )
            
            # Remove batch dimension
            node_features = node_features.squeeze(0)  # [max_num_segments, max_segment_len, output_dim]
            
        # Ensure no gradient
        return node_features.detach()

    def get_graph_embedding(self, utterance_features, speaker_ids, attention_masks):
        """
        Get embedding representation for the entire dialogue graph (non-batched, builds graph internally).
        
        Parameters:
            utterance_features: Utterance features [batch_size, max_num_segments, max_segment_len, feat_dim]
            speaker_ids: Speaker IDs [batch_size, max_num_segments] or List[int]
            attention_masks: Attention masks [batch_size, max_num_segments, max_segment_len]
            
        Returns:
            Graph embedding representation [output_dim] (batch dimension removed)
        """
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            # Use forward method to process input
            # forward method now returns graph embedding by default
            graph_embedding = self.forward(
                utterance_features,
                speaker_ids,
                attention_masks=attention_masks
            )
            
            # Remove batch dimension
            graph_embedding = graph_embedding.squeeze(0) # [output_dim]
            
        # Ensure no gradient
        return graph_embedding.detach()