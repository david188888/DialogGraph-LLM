import os
from pathlib import Path
from batch_transcribe import BatchTranscriber
from config import XUNFEI_APPID, XUNFEI_SECRET_KEY

def test_batch_transcribe():
    try:
        # 设置相关路径
        base_dir = Path(__file__).parent.parent
        audio_dir = base_dir / "data" / "audio_mp3"
        split_dir = base_dir / "data" / "split_audio"
        output_csv = base_dir / "data" / "transcriptions" / "segments.csv"

        # 确保目录存在
        split_dir.mkdir(exist_ok=True)
        output_csv.parent.mkdir(exist_ok=True)

        # 创建转写器实例，设置较小的并发数进行测试
        transcriber = BatchTranscriber(XUNFEI_APPID, XUNFEI_SECRET_KEY, max_workers=2)

        print("\n=== 开始批量转写测试 ===")
        print(f"输入目录: {audio_dir}")
        print(f"切分音频输出目录: {split_dir}")
        print(f"CSV输出路径: {output_csv}")

        # 获取前3个音频文件进行测试
        test_files = list(audio_dir.glob("*.mp3"))[:3]
        test_dir = audio_dir / "test_subset"
        test_dir.mkdir(exist_ok=True)

        # 创建测试子集
        for file in test_files:
            os.symlink(file, test_dir / file.name)

        print(f"\n将使用 {len(test_files)} 个文件进行测试:")
        for file in test_files:
            print(f"- {file.name}")

        # 执行批量转写
        results = transcriber.process_directory(
            str(test_dir),
            str(split_dir),
            str(output_csv)
        )

        # 输出处理统计
        success = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')

        print(f"\n=== 测试完成 ===")
        print(f"成功：{success}个文件")
        print(f"失败：{failed}个文件")

        # 输出失败的文件
        if failed > 0:
            print("\n失败的文件：")
            for result in results:
                if result['status'] == 'error':
                    print(f"- {result['file']}: {result['error']}")

        # 检查输出文件
        if os.path.exists(output_csv):
            print(f"\nCSV文件已生成: {output_csv}")
            print("前5行内容:")
            with open(output_csv, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 5:  # 只显示前5行
                        print(line.strip())
                    else:
                        break
        
        # 清理测试目录
        for file in test_dir.iterdir():
            os.unlink(file)
        test_dir.rmdir()

    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    test_batch_transcribe()