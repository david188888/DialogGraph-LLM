import os
import shutil
from pathlib import Path

def sort_audio_files():
    # 定义路径
    audio_dir = Path('data/audio')  # 修改为正确的路径
    mp3_dir = Path('data/audio_mp3')
    wav_dir = Path('data/audio')
    
    # 创建mp3文件夹（如果不存在）
    mp3_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历audio文件夹中的所有文件
    for file_path in audio_dir.glob('*'):
        if file_path.is_file():
            # 检查文件扩展名
            if file_path.suffix.lower() == '.mp3':
                # 移动mp3文件到audio_mp3文件夹
                shutil.move(str(file_path), str(mp3_dir / file_path.name))
                print(f"已移动 {file_path.name} 到 audio_mp3 文件夹")
            elif file_path.suffix.lower() == '.wav':
                # wav文件保持在原文件夹
                print(f"{file_path.name} 保留在 audio 文件夹")

if __name__ == '__main__':
    sort_audio_files()
    print("音频文件分类完成！")
