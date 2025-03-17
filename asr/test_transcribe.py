import os
import json
from pathlib import Path
from iflytek_asr import IflytekASR
from config import XUNFEI_APPID, XUNFEI_SECRET_KEY

# 讯飞开放平台配置信息 


def test_single_file():
    try:
        # 设置相关路径
        base_dir = Path(__file__).parent.parent
        audio_dir = base_dir / "data" / "audio"
        result_dir = base_dir / "data" / "transcriptions"
        result_dir.mkdir(exist_ok=True)

        # 获取第一个wav文件
        wav_files = list(audio_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"在 {audio_dir} 中未找到WAV文件")
            
        test_file = wav_files[2]
        print(f"\n=== 开始测试 ===")
        print(f"测试文件: {test_file}")
        print(f"文件大小: {os.path.getsize(test_file)} 字节")
        
        # 创建ASR实例并执行转写
        asr = IflytekASR(XUNFEI_APPID, XUNFEI_SECRET_KEY)
        print("\n开始转写流程...")
        result = asr.transcribe(str(test_file))['result']
        
        # 只保存 lattice 内容到JSON文件
        audio_id = test_file.stem
        json_path = result_dir / f"{audio_id}.json"
        print(f"\n保存ASR结果到: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({'lattice': result['lattice']}, f, ensure_ascii=False, indent=2)

        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    test_single_file()