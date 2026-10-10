import os

# Numba 和底层的并发库在某些环境下可能会导致段错误，强制它们使用单线程模式
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-small")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "bert-base-chinese")

import uuid
import numpy as np
import subprocess
import torch
import onnxruntime as ort
import librosa
from pathlib import Path
from transformers import pipeline
from MSA_FET import FeatureExtractionTool


# --- 辅助函数：对齐 ---
def pad_or_truncate(arr, target_len):
    curr_len, dim = arr.shape
    if curr_len > target_len:
        return arr[:target_len, :]  
    else:
        padding = np.zeros((target_len - curr_len, dim))
        return np.vstack((arr, padding))

class EmotionEngine:
    """多模态情感分析引擎（全局单例，只在启动时加载一次）"""
    def __init__(self, onnx_path):
        print("[Engine] 初始化开始...")

        # 自动获取当前用户家目录，强行创建 MMSA-FET 依赖的 tmp 和 log 目录
        # 这样无论是 Docker (家目录为 /root) 还是本地 WSL，都能防止 ffmpeg 写入时因找不到文件夹报错
        try:
            mmsa_base = Path.home() / ".MMSA-FET"
            mmsa_tmp = mmsa_base / "tmp"
            mmsa_log = mmsa_base / "log"
            
            mmsa_tmp.mkdir(parents=True, exist_ok=True)
            mmsa_log.mkdir(parents=True, exist_ok=True)
            print(f"[Engine] 依赖环境目录校验成功 ✅: {mmsa_tmp}")
        except Exception as dir_err:
            print(f"⚠️ 创建 MMSA-FET 依赖目录时发生异常 (将尝试继续运行): {dir_err}")
        
        # 1. 初始化 ONNX 推理器
        print(f"[Engine] 正在加载情感分析 ONNX 模型: {onnx_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_nodes = self.session.get_inputs()
        
        # 2. 初始化 ASR (Whisper) 并驻留显存
        print(f"[Engine] 正在加载 Whisper ASR 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_pipeline = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device=self.device)
        
        # 3. 初始化 MMSA-FET 工具
        print(f"[Engine] 正在初始化 MMSA-FET 特征提取工具...")
        self.config = {
            "text": {"model": "bert", "device": self.device, "pretrained": TEXT_MODEL, "args": {}},
            "audio": {"tool": "librosa", "sample_rate": 22050, "features": {"mfcc": {"n_mfcc": 20, "htk": True}, "chroma_cqt": {"n_chroma": 12}, "log_f0": {}}},
            "video": {"tool": "openface", "fps": 30, "average_over": 1, "args": {"landmark_2D": True, "landmark_3D": True, "head_pose": True, "action_units": True, "gaze": True, "pdmparams": True, "hogalign": False, "simalign": False, "nobadaligned": False, "tracked": False}}
        }
        self.fet = FeatureExtractionTool(config=self.config)

        # 4. Numba 暖机逻辑
        print("[Engine] 正在进行 Numba 暖机（强制触发音频算法编译以避免高并发冲突）...")
        try:
            # 制造 1 秒钟的随机音频数据
            dummy_signal = np.random.randn(22050).astype(np.float32)
            # 提前调用音频处理模块，强制 Numba 在主线程启动时安全地完成编译 [3]
            _ = librosa.feature.mfcc(y=dummy_signal, sr=22050, n_mfcc=20, htk=True)
            _ = librosa.feature.chroma_cqt(y=dummy_signal, sr=22050, n_chroma=12)
            print("[Engine] Numba 暖机完成！✅")
        except Exception as warm_err:
            print(f"⚠️ Numba 暖机发生异常（将尝试继续运行）: {warm_err}")
        
        print("[Engine] 初始化全部完成！✅")

    def _run_asr(self, video_path, req_id):
        """内部运行 ASR 提取"""
        tmp_wav = f"tmp_asr_{req_id}.wav" # 使用请求ID防止多人访问覆盖同一文件
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", tmp_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 强制中文
        result = self.asr_pipeline(tmp_wav, generate_kwargs={"language": "chinese", "task": "transcribe"})
        text = result["text"].strip()
        
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
            
        return text if text else "啊" # 兜底策略

    def process_video(self, video_path):
        """对外暴露的主处理方法"""
        req_id = str(uuid.uuid4())[:8] # 分配请求唯一标识
        
        # 1. 运行 ASR
        print(f"[{req_id}] 开始提取文本...")
        text = self._run_asr(video_path, req_id)
        
        # 2. 提取特征
        print(f"[{req_id}] 开始 MMSA-FET 提取特征 (文本识别为: {text})...")
        res = self.fet.run_single(in_file=video_path, text=text, return_type='np')
        
        # 3. 对齐
        print(f"[{req_id}] 执行特征截断与对齐...")
        res['audio'] = pad_or_truncate(res['audio'], 400)
        res['vision'] = pad_or_truncate(res['vision'], 55)
        res['text'] = pad_or_truncate(res['text'], 39)
        
        raw_text_bert = res['text_bert']
        padded_text_bert = np.zeros((3, 39))
        copy_len = min(raw_text_bert.shape[1], 39)
        padded_text_bert[:, :copy_len] = raw_text_bert[:, :copy_len]
        res['text_bert'] = padded_text_bert
        
        # 4. ONNX 推理
        print(f"[{req_id}] 拼装输入，执行 ONNX 推理...")
        v = np.expand_dims(np.array(res['vision'], dtype=np.float32), axis=0) 
        a = np.expand_dims(np.array(res['audio'], dtype=np.float32), axis=0) 
        t = np.expand_dims(np.array(res['text_bert'], dtype=np.float32), axis=0) 
        input_feed = {self.input_nodes[0].name: v, self.input_nodes[1].name: a, self.input_nodes[2].name: t}
        ort_outs = self.session.run(None, input_feed)
        sentiment_score = float(np.squeeze(ort_outs[0]))
        
        polarity = "Neutral"
        if sentiment_score > 0.1: polarity = "Positive"
        elif sentiment_score < -0.1: polarity = "Negative"
            
        return text, sentiment_score, polarity
