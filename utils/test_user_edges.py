import torch
from graph import DialogueGraphBuilder, DialogueGraph, Edge

def test_user_context_edges():
    """测试用户关联边的连接逻辑"""
    print("\n=== 测试用户关联边连接逻辑 ===")
    
    # 创建简单的测试用例
    speaker_ids = [0, 1, 0, 1, 0, 0, 1]
    num_nodes = len(speaker_ids)
    context_window_size = 2
    
    print(f"说话者ID序列: {speaker_ids}")
    print(f"上下文窗口大小: {context_window_size}")
    
    # 手动构建图
    graph = DialogueGraph()
    
    # 添加节点
    for i in range(num_nodes):
        node = Node(node_id=i, utterance_features=None, speaker_id=speaker_ids[i], timestamp=i)
        graph.add_node(node)
    
    # 构建用户关联边
    graph.build_user_context_edges(context_window_size)
    
    # 打印所有边
    print("\n所有用户关联边:")
    for edge in graph.edges:
        if edge.edge_type == Edge.USER_CONTEXT_EDGE:
            print(f"  节点{edge.source_id}(说话者{speaker_ids[edge.source_id]}) -> "
                  f"节点{edge.target_id}(说话者{speaker_ids[edge.target_id]})")
    
    # 检查特定的连接
    # 最后一个节点(6)应该和第四个节点(3)连接，因为它们都是说话者1
    has_connection = False
    for edge in graph.edges:
        if (edge.edge_type == Edge.USER_CONTEXT_EDGE and
            edge.source_id == 3 and edge.target_id == 6):
            has_connection = True
            break
    
    if has_connection:
        print("\n✅ 测试通过：最后一个节点(6)与第四个节点(3)正确连接")
    else:
        print("\n❌ 测试失败：最后一个节点(6)与第四个节点(3)未连接")
    
    # 验证窗口大小扩大后的连接
    print("\n检查窗口大小=2时的连接:")
    # 说话者0的节点分布在0，2，4，5位置
    # 窗口大小为2时，节点5应该连接到节点4和节点2，但不连接到节点0
    has_connection_5_4 = False
    has_connection_5_2 = False
    has_connection_5_0 = False
    
    for edge in graph.edges:
        if edge.edge_type == Edge.USER_CONTEXT_EDGE and edge.target_id == 5:
            if edge.source_id == 4:
                has_connection_5_4 = True
            elif edge.source_id == 2:
                has_connection_5_2 = True
            elif edge.source_id == 0:
                has_connection_5_0 = True
    
    print(f"  节点5与节点4连接: {'✅' if has_connection_5_4 else '❌'}")
    print(f"  节点5与节点2连接: {'✅' if has_connection_5_2 else '❌'} (应该连接，在窗口范围内)")
    print(f"  节点5与节点0连接: {'❌' if not has_connection_5_0 else '✅'} (应该不连接，超出窗口)")

    # 节点6(最后一个)还应该与节点1(第二个)连接，因为窗口扩大到2
    has_connection_6_1 = False
    
    for edge in graph.edges:
        if (edge.edge_type == Edge.USER_CONTEXT_EDGE and 
            edge.source_id == 1 and edge.target_id == 6):
            has_connection_6_1 = True
            break
    
    print(f"  节点6与节点1连接: {'✅' if has_connection_6_1 else '❌'} (窗口=2时应该连接)")
    
    return has_connection

if __name__ == "__main__":
    from graph import Node  # 在这里导入避免循环导入问题
    test_user_context_edges() 