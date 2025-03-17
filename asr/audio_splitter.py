from pydub import AudioSegment
import json

class AudioSplitter:
    def __init__(self):
        """初始化音频切分器"""
        pass

    def _extract_text_from_lattice(self, json_1best_str) -> tuple:
        """从lattice格式中提取完整的文本和说话人信息
        :param json_1best_str: lattice中json_1best字段内容
        :return: (文本, 说话人, 开始时间, 结束时间)的元组
        """
        # 初始化变量
        text = ""
        speaker = ""
        start_time = 0
        end_time = 0
        
        # 如果是字符串，先转换成字典
        if isinstance(json_1best_str, str):
            try:
                json_1best = json.loads(json_1best_str)
            except:
                return "", "", 0, 0
        else:
            json_1best = json_1best_str

        if not isinstance(json_1best, dict):
            return "", "", 0, 0
            
        # 解析st字段
        if 'st' in json_1best:
            st = json_1best['st']
            # 获取说话人标识
            speaker = st.get('rl', '')
            # 获取时间信息
            start_time = float(st.get('bg', '0')) / 1000  # 转换为秒
            end_time = float(st.get('ed', '0')) / 1000
            
            # 从rt字段提取文本内容
            if 'rt' in st and len(st['rt']) > 0:
                for ws in st['rt'][0].get('ws', []):
                    if 'cw' in ws:
                        for cw in ws['cw']:
                            word = cw.get('w', '').strip()
                            word_type = cw.get('wp', 'n')
                            if word and word_type in ['n', 'p']:
                                text += word
                                
        return text, speaker, start_time, end_time

    def parse_xunfei_json(self, json_result) -> list:
        """解析讯飞ASR的JSON结果
        :param json_result: 讯飞ASR返回的JSON结果
        :return: 列表，包含每个语音片段的信息(起止时间、说话人、文本)
        """
        segments = []
        if 'lattice' not in json_result:
            return segments
            
        for item in json_result['lattice']:
            # 从json_1best提取文本和说话人信息
            text, speaker, start_time, end_time = self._extract_text_from_lattice(item['json_1best'])
            
            if text:  # 只添加有文本内容的片段
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker': speaker,
                    'text': text
                })
            
        return segments

    def split_audio(self, audio_path: str, output_path: str, start_time: float, end_time: float):
        """切分单个音频片段
        :param audio_path: 输入音频文件路径
        :param output_path: 输出音频文件路径
        :param start_time: 开始时间(秒)
        :param end_time: 结束时间(秒)
        """
        # 加载音频文件
        # audio = AudioSegment.from_wav(audio_path)
        audio = AudioSegment.from_mp3(audio_path)

        # 转换时间为毫秒
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # 切分音频
        segment_audio = audio[start_ms:end_ms]
        
        # 保存切分后的音频
        segment_audio.export(output_path, format="wav")