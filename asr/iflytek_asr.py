import base64
import hashlib
import hmac
import json
import os
import time
import requests
import urllib
from typing import Dict

class IflytekASR:
    def __init__(self, appid: str, secret_key: str):
        self.appid = appid
        self.secret_key = secret_key
        self.host = 'https://raasr.xfyun.cn/v2/api'
        self.upload_url = '/upload'
        self.get_result_url = '/getResult'
        self.ts = str(int(time.time()))
        self.signa = self._get_signa()

    def _get_signa(self) -> str:
        """生成签名"""
        m2 = hashlib.md5()
        m2.update((self.appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(self.secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        return str(signa, 'utf-8')

    def upload_audio(self, audio_path: str) -> dict:
        """上传音频文件"""
        # print("上传部分：")
        file_len = os.path.getsize(audio_path)
        file_name = os.path.basename(audio_path)
        
        param_dict = {
            'appId': self.appid,
            'signa': self.signa,
            'ts': self.ts,
            'fileSize': file_len,
            'fileName': file_name,
            'resultType': 'transfer',
            'language': 'cn',
            'roleType': 1,
            'roleNum': 2,
            'duration': "200",
            # 'trackMode': 2
        }
        
        # print("upload参数：", param_dict)
        with open(audio_path, 'rb') as f:
            data = f.read(file_len)
        
        response = requests.post(
            url=f"{self.host}{self.upload_url}?{urllib.parse.urlencode(param_dict)}",
            headers={"Content-type": "application/json"},
            data=data
        )
        # print("upload_url:", response.request.url)
        result = json.loads(response.text)
        # print("upload resp:", result)
        
        return result

    def get_result(self, order_id: str = None) -> dict:
        """获取转写结果"""
        if order_id is None:
            upload_result = self.upload_audio(self.current_file)
            order_id = upload_result['content']['orderId']
        
        param_dict = {
            'appId': self.appid,
            'signa': self.signa,
            'ts': self.ts,
            'orderId': order_id,
            'resultType': "transfer,predict"
        }
        
        # print("\n查询部分：")
        # print("get result参数：", param_dict)
        
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(
                url=f"{self.host}{self.get_result_url}?{urllib.parse.urlencode(param_dict)}",
                headers={"Content-type": "application/json"}
            )
            result = json.loads(response.text)
            # print(result)
            status = result['content']['orderInfo']['status']
            # print("status=", status)
            if status == 4:
                break
            time.sleep(5)
            
        # print("get_result resp:", result)
        return result

    def transcribe(self, audio_path: str) -> Dict:
        """完整的音频转写流程"""
        self.current_file = audio_path
        try:
            # 上传并获取结果
            result = self.get_result()
            
            # 解析结果
            order_result = json.loads(result['content']['orderResult'])
            
            return {
                'status': 'success',
                'result': order_result
            }
            
        except Exception as e:
            print(f"转写过程中发生错误：{str(e)}")
            raise