import os
from pathlib import Path
import librosa
import pandas as pd

def check_audio_channels():
    # 设置音频文件夹路径
    base_dir = Path(__file__).parent
    audio_dir = base_dir / "data" / "audio_mp3"
    
    if not audio_dir.exists():
        print(f"错误: 音频文件夹 {audio_dir} 不存在")
        return
    
    # 存储结果
    results = []
    
    # 遍历所有音频文件
    for audio_file in audio_dir.glob("*.mp3"):
        try:
            # 加载音频文件
            y, sr = librosa.load(str(audio_file), mono=False)
            
            # 检查声道数
            channels = 1 if y.ndim == 1 else y.shape[0]
            
            results.append({
                "文件名": audio_file.name,
                "声道数": channels,
                "类型": "单声道" if channels == 1 else "双声道"
            })
                
        except Exception as e:
            print(f"处理文件 {audio_file.name} 时发生错误: {str(e)}")
    
    # 转换为DataFrame以便更好地展示
    df = pd.DataFrame(results)
    
    # 按声道数分组统计
    print("\n=== 音频文件声道统计 ===")
    print(df["类型"].value_counts())
    
    print("\n=== 详细信息 ===")
    # print(df.to_string(index=False))
    
    # 保存结果到CSV文件
    output_file = base_dir / "audio_channels_report.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    check_audio_channels()