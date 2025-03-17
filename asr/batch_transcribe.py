import os
import csv
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from iflytek_asr import IflytekASR
from audio_splitter import AudioSplitter
from config import XUNFEI_APPID, XUNFEI_SECRET_KEY

class BatchTranscriber:
    def __init__(self, appid: str, secret_key: str, max_workers: int = 3):
        """
        初始化批量转写器
        :param appid: 讯飞API的APPID
        :param secret_key: 讯飞API的密钥
        :param max_workers: 最大并行处理数量
        """
        self.asr = IflytekASR(appid, secret_key)
        self.splitter = AudioSplitter()
        self.max_workers = max_workers
        # 使用信号量控制并发请求数
        self.semaphore = threading.Semaphore(15)
        # 用于存储所有片段信息的列表
        self.all_segments = []
        self.segments_lock = threading.Lock()

    def process_single_file(self, audio_path: str, split_dir: str) -> Dict:
        """处理单个音频文件，包括ASR和音频切分"""
        try:
            with self.semaphore:  # 控制并发请求数
                # 执行ASR
                result = self.asr.transcribe(audio_path)
                if result['status'] != 'success':
                    raise Exception(f"ASR failed for {audio_path}")

                # 解析ASR结果获取时间片段
                segments = self.splitter.parse_xunfei_json(result['result'])
                
                # 音频ID（文件名）
                audio_id = Path(audio_path).stem
                file_segments = []

                # 处理每个片段
                for i, segment in enumerate(segments, 1):
                    # 构建片段ID和输出路径
                    segment_id = f"{audio_id}_{i}"
                    output_path = os.path.join(split_dir, f"{segment_id}.wav")
                    
                    # 切分音频
                    self.splitter.split_audio(
                        audio_path, 
                        output_path, 
                        segment['start_time'], 
                        segment['end_time']
                    )
                    
                    # 保存片段信息
                    segment_info = {
                        'audio_id': segment_id,
                        'text': segment['text'],
                        'speaker': segment['speaker'],
                        'start_time': f"{segment['start_time']:.1f}",
                        'end_time': f"{segment['end_time']:.1f}"
                    }
                    file_segments.append(segment_info)

                # 线程安全地添加片段信息
                with self.segments_lock:
                    self.all_segments.extend(file_segments)

                return {
                    'status': 'success',
                    'file': audio_path,
                    'segments_count': len(file_segments)
                }

        except Exception as e:
            return {
                'status': 'error',
                'file': audio_path,
                'error': str(e)
            }

    def process_directory(self, 
                         input_dir: str, 
                         split_dir: str,
                         output_csv: str,
                         file_types: List[str] = None) -> List[Dict]:
        """
        处理目录中的所有音频文件
        :param input_dir: 输入音频文件目录
        :param split_dir: 切分后音频保存目录
        :param output_csv: 输出CSV文件路径
        :param file_types: 要处理的文件类型列表，默认为['.wav', '.mp3']
        :return: 处理结果列表
        """
        if file_types is None:
            file_types = ['.wav', '.mp3']

        # 创建输出目录
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # 获取所有音频文件
        audio_files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if any(f.lower().endswith(ext) for ext in file_types)
        ]

        results = []
        # 清空片段信息列表
        self.all_segments = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_single_file, 
                    audio_file, 
                    split_dir
                ): audio_file 
                for audio_file in audio_files
            }

            # 使用tqdm显示进度
            with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    # 显示当前文件处理状态
                    if result['status'] == 'success':
                        pbar.set_postfix(
                            file=os.path.basename(result['file']),
                            segments=result['segments_count']
                        )
                    else:
                        pbar.set_postfix(
                            file=os.path.basename(result['file']),
                            error=result['error']
                        )

        # 保存所有片段信息到CSV
        if self.all_segments:
            with open(output_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=['audio_id', 'text', 'speaker', 'start_time', 'end_time']
                )
                writer.writeheader()
                writer.writerows(self.all_segments)

        return results

def main():
    # 配置参数
    base_dir = Path(__file__).parent.parent
    INPUT_DIR = base_dir / "data" / "audio_mp3"
    SPLIT_DIR = base_dir / "data" / "split_audio"
    OUTPUT_CSV = base_dir / "data" / "transcriptions" / "segments.csv"
    
    # 创建转写器实例
    transcriber = BatchTranscriber(XUNFEI_APPID, XUNFEI_SECRET_KEY)
    
    # 处理音频文件
    results = transcriber.process_directory(INPUT_DIR, SPLIT_DIR, OUTPUT_CSV)
    
    # 输出处理统计
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n处理完成！")
    print(f"成功：{success}个文件")
    print(f"失败：{failed}个文件")
    
    # 输出失败的文件
    if failed > 0:
        print("\n失败的文件：")
        for result in results:
            if result['status'] == 'error':
                print(f"- {result['file']}: {result['error']}")

if __name__ == '__main__':
    main()