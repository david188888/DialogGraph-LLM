import os
import json
import csv
from pathlib import Path
from audio_splitter import AudioSplitter
from collections import defaultdict

def parse_json_1best(json_1best):
    """解析json_1best获取文本内容
    :param json_1best: JSON字符串或字典
    :return: 文本内容
    """
    # 如果是字符串，先转换成字典
    if isinstance(json_1best, str):
        try:
            json_1best = json.loads(json_1best)
        except:
            return ""
    
    text = ""
    # 检查st和rt字段是否存在
    if isinstance(json_1best, dict) and 'st' in json_1best and 'rt' in json_1best['st']:
        for rt_item in json_1best['st']['rt']:
            for ws in rt_item['ws']:
                for cw in ws['cw']:
                    if 'w' in cw and cw['w'].strip():
                        # 只处理正常词(n)和标点(p)，跳过顺滑词(s)
                        if 'wp' not in cw or cw['wp'] in ['n', 'p']:
                            text += cw['w']
    return text

def test_audio_split():
    try:
        # 设置相关路径
        base_dir = Path(__file__).parent.parent
        transcription_dir = base_dir / "data" / "transcriptions"
        split_dir = base_dir / "data" / "split_audio"
        split_dir.mkdir(exist_ok=True)
        
        # 获取第一个JSON文件
        json_files = list(transcription_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"在 {transcription_dir} 中未找到JSON文件")
            
        test_file = json_files[0]
        print(f"\n=== 开始测试 ===")
        print(f"测试文件: {test_file}")
        
        # 读取ASR结果
        with open(test_file, "r", encoding="utf-8") as f:
            result = json.load(f)
            
        # 创建AudioSplitter实例
        splitter = AudioSplitter()
        
        # 准备音频文件路径
        audio_id = test_file.stem
        audio_path = base_dir / "data" / "audio" / f"{audio_id}.wav"
        
        # 检查音频文件是否存在
        if not audio_path.exists():
            raise FileNotFoundError(f"未找到对应的音频文件: {audio_path}")
        
        print("\n开始切分音频...")
        # 解析JSON获取片段信息
        segments = splitter.parse_xunfei_json(result)
        
        # 全局片段计数器
        segment_counter = 0
        
        csv_segments = []
        # 处理每个切分片段
        for i, segment in enumerate(segments, 1):
            start_time = segment['start_time']
            end_time = segment['end_time']
            speaker = segment['speaker']
            text = segment['text']
            
            # 输出当前处理信息
            print(f"\n[片段 {i}] {speaker}")
            print(f"时间: {start_time:.1f}s - {end_time:.1f}s")
            print(f"文本: {text}")
            
            # 更新全局计数器
            segment_counter += 1
            
            # 构建人性化的audio_id，只包含电话号码和序号
            human_readable_id = f"{audio_id}_{segment_counter}"
            
            # 构建音频文件名，与audio_id保持一致
            output_file = split_dir / f"{human_readable_id}.wav"
            
            # 切分音频
            splitter.split_audio(str(audio_path), str(output_file), start_time, end_time)
            
            # 记录片段信息用于CSV
            csv_segments.append({
                'audio_id': human_readable_id,
                'text': text,
                'speaker': speaker,
                'start_time': f"{start_time:.1f}",
                'end_time': f"{end_time:.1f}"
            })

        # 保存到CSV文件
        csv_path = transcription_dir / "segments.csv"
        print(f"\n保存片段信息到: {csv_path}")
        
        # 写入CSV文件
        with open(csv_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['audio_id', 'text', 'speaker', 'start_time', 'end_time'])
            writer.writeheader()
            writer.writerows(csv_segments)

        print(f"\n=== 测试完成 ===")
        print(f"已生成 {len(csv_segments)} 个音频片段")
        print(f"音频片段保存在: {split_dir}")
        print(f"文本信息保存在: {csv_path}")
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    test_audio_split()