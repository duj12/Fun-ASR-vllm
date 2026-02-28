#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版音频处理器
支持多种VAD模型进行音频切分，对切分后的音频和原始短音频进行转写
结果保存到Excel文件中
"""

import os
import sys
import json
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import re
from pathlib import Path
from typing import List, Dict, Optional, Union
from qwen_asr import Qwen3ASRModel
from funasr import AutoModel

# 配置日志
logging.root.handlers = []  # 清空modelscope修改后的handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.FileHandler('simple_audio_processing.log', encoding='utf-8'),
    #     logging.StreamHandler()
    # ]
)
logger = logging.getLogger(__name__)

CUR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CUR_DIR, "FireRedASR2S"))
# FireRedVad 相关导入（可选）
try:
    from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
    FIRERED_VAD_AVAILABLE = True
except ImportError:
    FIRERED_VAD_AVAILABLE = False
    logger.warning("FireRedVad 未安装，将使用默认的 fsmn-vad 模型")


class VADModelWrapper:
    """VAD模型包装器，支持多种VAD模型"""
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.model_type = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化VAD模型"""
        if self.model_name.lower() == "fireredvad":
            if not FIRERED_VAD_AVAILABLE:
                raise ImportError("FireRedVad 模型不可用，请安装 FireRedASR2S 库")
            
            logger.info("正在加载 FireRedVad 模型")
            vad_config = FireRedVadConfig(
                use_gpu=self.device.startswith("cuda"),
                smooth_window_size=5,
                speech_threshold=0.4,
                min_speech_frame=20,
                max_speech_frame=2000,
                min_silence_frame=20,
                merge_silence_frame=0,
                extend_speech_frame=0,
                chunk_max_frame=30000
            )
            self.model = FireRedVad.from_pretrained(
                "FireRedASR2S/pretrained_models/FireRedVAD/VAD", 
                vad_config
            )
            self.model_type = "fireredvad"
            logger.info("FireRedVad 模型加载完成")
            
        else:
            # 默认使用 fsmn-vad
            logger.info(f"正在加载VAD模型: {self.model_name}")
            self.model = AutoModel(model=self.model_name, model_revision="v2.0.4")
            self.model_type = "fsmnvad"
            logger.info(f"{self.model_name} 模型加载完成")
    
    def detect_segments(self, audio_path: str) -> List[Dict]:
        """
        检测音频中的语音段
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[Dict]: 语音段信息列表，每个元素包含 start_time, end_time, duration
        """
        if self.model_type == "fireredvad":
            return self._detect_with_fireredvad(audio_path)
        else:
            return self._detect_with_fsmnvad(audio_path)
    
    def _detect_with_fireredvad(self, audio_path: str) -> List[Dict]:
        """使用 FireRedVad 检测语音段"""
        try:
            result, probs = self.model.detect(audio_path)
            
            segments = []
            if 'timestamps' in result:
                for i, (start_sec, end_sec) in enumerate(result['timestamps']):
                    segments.append({
                        'segment_id': i,
                        'start_time': start_sec * 1000,  # 转换为毫秒
                        'end_time': end_sec * 1000,      # 转换为毫秒
                        'duration': (end_sec - start_sec) * 1000  # 转换为毫秒
                    })
            
            logger.info(f"FireRedVad 切分完成，共获得 {len(segments)} 个语音段")
            return segments
            
        except Exception as e:
            logger.error(f"FireRedVad 切分失败: {e}")
            return []
    
    def _detect_with_fsmnvad(self, audio_path: str) -> List[Dict]:
        """使用 fsmn-vad 检测语音段"""
        try:
            vad_result = self.model.generate(input=audio_path)
            
            segments = []
            if vad_result and len(vad_result) > 0:
                for i, segment_info in enumerate(vad_result[0]['value']):
                    segments.append({
                        'segment_id': i,
                        'start_time': segment_info[0],
                        'end_time': segment_info[1],
                        'duration': segment_info[1] - segment_info[0]
                    })
            
            logger.info(f"fsmn-vad 切分完成，共获得 {len(segments)} 个语音段")
            return segments
            
        except Exception as e:
            logger.error(f"fsmn-vad 切分失败: {e}")
            return []


class AudioFilter:
    """音频筛选器，用于过滤低质量的转写结果"""
    
    def __init__(self, 
                 min_chars_no_punct: int = 3,
                 similarity_threshold: float = 0.6,
                 noise_indicators: List[str] = None):
        """
        初始化筛选器
        
        Args:
            min_chars_no_punct: 无标点时的最小字符数
            similarity_threshold: 文本相似度阈值（用于检测重复内容）
            noise_indicators: 噪声指示词列表
        """
        self.min_chars_no_punct = min_chars_no_punct
        self.similarity_threshold = similarity_threshold
        self.noise_indicators = noise_indicators or [
            '嗯', '啊', '呃', '哦', '哈', '嘿', '哼', '咳', '喂'
        ]
        
        # 中文标点符号
        self.chinese_punctuation = set(',.?!:，。！？；：""''（）【】《》、')
    
    def remove_punctuation(self, text: str) -> str:
        """移除文本中的标点符号"""
        if not text:
            return ""
        # 移除中文标点
        for punct in self.chinese_punctuation:
            text = text.replace(punct, '')
        # 移除英文标点
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（基于字符级别的Jaccard相似度）
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        # 转换为字符集合
        set1 = set(text1)
        set2 = set(text2)
        
        # 计算交集和并集
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if len(union) == 0:
            return 0.0
            
        return len(intersection) / len(union)
    
    def is_noise_text(self, text: str) -> bool:
        """
        判断文本是否为噪声文本
        
        Args:
            text: 待判断的文本
            
        Returns:
            bool: 是否为噪声文本
        """
        if not text:
            return True
            
        clean_text = self.remove_punctuation(text)
        
        # 规则1: 无标点且字符数过少
        if len(clean_text) <= self.min_chars_no_punct:
            return True
            
        # 规则2: 包含过多噪声词汇
        noise_words = [word for word in self.noise_indicators if word in text]
        if len(noise_words) > 0 and len(clean_text) <= 5:
            # 如果包含噪声词且总长度很短，则认为是噪声
            noise_ratio = len(''.join(noise_words)) / len(clean_text)
            if noise_ratio > 0.5:
                return True
                
        # 规则3: 重复字符过多（可能是噪声）
        if len(set(clean_text)) < len(clean_text) * 0.3:  # 字符多样性不足30%
            return True
            
        return False
    
    def filter_results(self, results: List[Dict]) -> List[Dict]:
        """
        对转写结果进行筛选
        
        Args:
            results: 转写结果列表
            
        Returns:
            List[Dict]: 筛选后的结果列表
        """
        logger.info(f"开始筛选 {len(results)} 条转写结果...")
        
        # 第一步：基础筛选 - 移除明显噪声的音频
        filtered_results = []
        removed_count = 0
        
        for result in results:
            if not self.is_noise_text(result['transcription']):
                filtered_results.append(result)
            else:
                logger.debug(f"移除噪声音频: {result['audio_name']} - '{result['transcription']}'")
                removed_count += 1
                # 删除对应的音频文件
                self._remove_audio_file(result)
        
        logger.info(f"基础筛选完成，移除 {removed_count} 条噪声结果")
        
        # 第二步：去重筛选 - 移除重复内容
        final_results = self._remove_duplicates(filtered_results)
        
        logger.info(f"去重筛选完成，最终保留 {len(final_results)} 条结果")
        
        return final_results
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """
        移除重复内容的音频（保留字数较多的）
        修改为不按类型分组，统一进行去重比较
        添加包含关系检测功能
        
        Args:
            results: 筛选后的结果列表
            
        Returns:
            List[Dict]: 去重后的结果列表
        """
        if len(results) <= 1:
            return results
            
        # 不再按类型分组，将所有音频放在一起进行去重比较
        # 这样可以发现VAD切分音频和原始短音频之间的重复内容
        
        # 按文本长度排序（长的优先保留）
        sorted_results = sorted(results, 
                              key=lambda x: len(self.remove_punctuation(x['transcription'])), 
                              reverse=True)
        
        kept_results = []
        removed_indices = set()
        
        for i, result in enumerate(sorted_results):
            if i in removed_indices:
                continue
                
            current_text = self.remove_punctuation(result['transcription'])
            current_full_text = result['transcription']  # 保留完整文本用于包含检测
            kept_results.append(result)
            
            # 检查所有后续结果是否与当前结果重复（包括不同类型）
            for j in range(i + 1, len(sorted_results)):
                if j in removed_indices:
                    continue
                    
                compare_result = sorted_results[j]
                compare_text = self.remove_punctuation(compare_result['transcription'])
                compare_full_text = compare_result['transcription']
                
                # 检查相似度
                similarity = self.calculate_similarity(current_text, compare_text)
                
                # 检查包含关系
                is_contained = self._is_text_contained(current_full_text, compare_full_text)
                
                if similarity >= self.similarity_threshold or is_contained:
                    reason = "包含关系" if is_contained else f"相似度{similarity:.2f}"
                    logger.debug(f"发现重复/包含内容，移除: {compare_result['audio_name']} "
                               f"(与 {result['audio_name']} {reason})")
                    removed_indices.add(j)
                    # 删除对应的音频文件
                    self._remove_audio_file(compare_result)
        
        # 对最终结果按音频名称排序，方便查看
        final_sorted_results = sorted(kept_results, key=lambda x: x['audio_name'])
        
        return final_sorted_results
    
    def _is_text_contained(self, longer_text: str, shorter_text: str) -> bool:
        """
        检查较短文本是否被较长文本完全包含
        
        Args:
            longer_text: 较长的文本
            shorter_text: 较短的文本
            
        Returns:
            bool: 是否存在包含关系
        """
        if not longer_text or not shorter_text:
            return False
            
        # 移除标点进行比较
        clean_longer = self.remove_punctuation(longer_text)
        clean_shorter = self.remove_punctuation(shorter_text)
        
        # 如果清理后的短文本长度大于长文本，肯定不包含
        if len(clean_shorter) > len(clean_longer):
            return False
            
        # 检查是否包含（忽略大小写）
        return clean_shorter.lower() in clean_longer.lower()
    
    def _deduplicate_group(self, group_results: List[Dict]) -> List[Dict]:
        """
        对同一组音频进行去重处理
        
        Args:
            group_results: 同一组的音频结果
            
        Returns:
            List[Dict]: 去重后的结果
        """
        if len(group_results) <= 1:
            return group_results
            
        # 按文本长度排序（长的优先保留）
        sorted_results = sorted(group_results, 
                              key=lambda x: len(self.remove_punctuation(x['transcription'])), 
                              reverse=True)
        
        kept_results = []
        removed_indices = set()
        
        for i, result in enumerate(sorted_results):
            if i in removed_indices:
                continue
                
            current_text = self.remove_punctuation(result['transcription'])
            kept_results.append(result)
            
            # 检查后续结果是否与当前结果重复
            for j in range(i + 1, len(sorted_results)):
                if j in removed_indices:
                    continue
                    
                compare_result = sorted_results[j]
                compare_text = self.remove_punctuation(compare_result['transcription'])
                
                similarity = self.calculate_similarity(current_text, compare_text)
                
                if similarity >= self.similarity_threshold:
                    logger.debug(f"发现重复内容，移除: {compare_result['audio_name']} "
                               f"(相似度: {similarity:.2f})")
                    removed_indices.add(j)
                    # 删除对应的音频文件
                    self._remove_audio_file(compare_result)
        
        return kept_results
    
    def _remove_audio_file(self, result: Dict):
        """
        删除对应的音频文件
        
        Args:
            result: 包含音频文件信息的结果字典
        """
        try:
            if 'audio_path' in result:
                audio_path = result['audio_path']
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.debug(f"已删除音频文件: {audio_path}")
        except Exception as e:
            logger.warning(f"删除音频文件失败: {e}")


class SimpleAudioProcessor:
    """简化版音频处理器"""
    
    def __init__(self, 
                 asr_model_name: str = "Qwen/Qwen3-ASR-1.7B",
                 vad_model_name: str = "fsmn-vad",
                 device: str = "cuda:0",
                 dtype_str: str = "bfloat16",
                 enable_filtering: bool = True,
                 min_chars_no_punct: int = 2,
                 similarity_threshold: float = 0.6):
        """
        初始化处理器
        """
        self.device = device
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.dtype = dtype_map.get(dtype_str, torch.bfloat16)
        
        # 初始化筛选器
        self.enable_filtering = enable_filtering
        if enable_filtering:
            self.audio_filter = AudioFilter(
                min_chars_no_punct=min_chars_no_punct,
                similarity_threshold=similarity_threshold
            )
        else:
            self.audio_filter = None
        
        # 初始化VAD模型包装器
        self.vad_wrapper = VADModelWrapper(vad_model_name, device)
        
        logger.info(f"正在加载ASR模型: {asr_model_name}")
        self.asr_model = Qwen3ASRModel.from_pretrained(
            asr_model_name,
            dtype=self.dtype,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
        
        logger.info("模型加载完成")
        
        # 统计信息
        self.stats = {
            'total_packages': 0,
            'processed_audios': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'vad_segments': 0,
            'empty_folders_removed': 0,
            'filtered_results': 0,  # 新增：筛选掉的结果数
            'final_results': 0      # 新增：最终保留的结果数
        }
    
    def load_pcm_audio(self, pcm_path: str) -> np.ndarray:
        """
        加载PCM音频文件
        
        Args:
            pcm_path: PCM文件路径
            
        Returns:
            np.ndarray: 音频数据
        """
        try:
            with open(pcm_path, 'rb') as f:
                raw_audio = np.frombuffer(f.read(), dtype=np.int16)
                audio_float = raw_audio.astype(np.float32) / 32768.0
            return audio_float
        except Exception as e:
            logger.error(f"加载PCM文件失败 {pcm_path}: {e}")
            raise
            
    def get_audio_duration(self, audio_data: np.ndarray, sample_rate: int = 16000) -> float:
        """
        计算音频时长（秒）
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            float: 音频时长（秒）
        """
        return len(audio_data) / sample_rate

    def vad_segmentation(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """
        使用配置的VAD模型进行音频切分
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            List[Dict]: 切分段信息列表
        """
        try:
            # 保存临时WAV文件供VAD模型使用
            temp_wav_path = "temp_vad_input.wav"
            sf.write(temp_wav_path, audio_data, sample_rate)
            
            # 使用VAD包装器进行切分
            segments = self.vad_wrapper.detect_segments(temp_wav_path)
            
            # 清理临时文件
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            
            logger.info(f"VAD切分完成，共获得 {len(segments)} 个语音段")
            self.stats['vad_segments'] += len(segments)
            
            return segments
            
        except Exception as e:
            logger.error(f"VAD切分失败: {e}")
            return []
    
    def extract_audio_segment(self, 
                            audio_data: np.ndarray,
                            start_time: float, 
                            end_time: float, 
                            sample_rate: int = 16000) -> np.ndarray:
        """
        提取音频片段
        
        Args:
            audio_data: 原始音频数据
            start_time: 开始时间(毫秒)
            end_time: 结束时间(毫秒)
            sample_rate: 采样率
            
        Returns:
            np.ndarray: 提取的音频片段
        """
        start_sample = int(start_time/1000.0 * sample_rate)
        end_sample = int(end_time/1000.0 * sample_rate)
        
        # 边界检查
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        return audio_data[start_sample:end_sample]
    
    def transcribe_audio(self, audio_path: str, language: str = "Chinese") -> str:
        """
        使用ASR模型对音频进行转写
        
        Args:
            audio_path: 音频文件路径
            language: 语言类型
            
        Returns:
            str: 转写结果文本
        """
        try:
            logger.info(f"正在转写音频: {audio_path}")
            results = self.asr_model.transcribe(
                audio=[audio_path],
                language=[language],
                return_time_stamps=False,
            )
            
            if results and len(results) > 0:
                transcription = results[0].text
                logger.info(f"转写完成: {transcription}")
                self.stats['successful_transcriptions'] += 1
                return transcription
            else:
                logger.warning(f"转写结果为空: {audio_path}")
                return ""
                
        except Exception as e:
            logger.error(f"ASR转写失败 {audio_path}: {e}")
            self.stats['failed_transcriptions'] += 1
            return ""
    
    def is_folder_empty(self, folder_path: str) -> bool:
        """
        检查文件夹是否为空（不包含WAV文件）
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            bool: 是否为空
        """
        wav_files = list(Path(folder_path).glob("*.wav"))
        return len(wav_files) == 0
    
    def remove_empty_folder(self, folder_path: str) -> bool:
        """
        删除空文件夹
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            bool: 是否成功删除
        """
        try:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # 检查是否真的为空（只有文件夹本身，没有WAV文件）
                if self.is_folder_empty(folder_path):
                    os.rmdir(folder_path)
                    logger.info(f"已删除空文件夹: {folder_path}")
                    self.stats['empty_folders_removed'] += 1
                    return True
            return False
        except Exception as e:
            logger.error(f"删除文件夹失败 {folder_path}: {e}")
            return False
    
    def save_folder_results_to_jsonl(self, results: List[Dict], folder_path: str, package_name: str):
        """
        将单个文件夹的转写结果保存为jsonl文件
        
        Args:
            results: 转写结果列表
            folder_path: 文件夹路径
            package_name: 包名
        """
        try:
            jsonl_path = os.path.join(folder_path, f"{package_name}_results.jsonl")
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for result in results:
                    # 构造jsonl记录
                    record = {
                        'audio_name': result['audio_name'],
                        'transcription': result['transcription'],
                        'audio_type': result['audio_type'],
                        'duration': result.get('duration', 0)  # 添加时长信息
                    }
                    
                    # 如果是VAD切分的音频，添加时间段信息
                    if result['audio_type'] == 'vad_segment' and 'segment_info' in result:
                        segment_info = result['segment_info']
                        record['segment_info'] = {
                            'segment_id': segment_info['segment_id'],
                            'start_time': segment_info['start_time'],
                            'end_time': segment_info['end_time'],
                            'duration': segment_info['duration']
                        }
                    
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"文件夹结果已保存到: {jsonl_path}")
            logger.info(f"共保存 {len(results)} 条记录")
            
        except Exception as e:
            logger.error(f"保存文件夹jsonl文件失败: {e}")
    
    def process_single_audio_package(self, 
                                   package_path: str, 
                                   output_base_dir: str) -> tuple[List[Dict], str]:
        """
        处理单个音频包
        
        Args:
            package_path: 压缩包路径
            output_base_dir: 输出基础目录
            
        Returns:
            tuple[List[Dict], str]: (转写结果列表, 处理后的文件夹路径)
        """
        package_name = Path(package_path).stem
        work_dir = os.path.join(output_base_dir, package_name)
        os.makedirs(work_dir, exist_ok=True)
        
        transcription_results = []
        folder_path = work_dir  # 保存文件夹路径用于后续处理
        
        try:
            # 解压文件
            import zipfile
            with zipfile.ZipFile(package_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    if member.filename.endswith('/'):
                        continue
                    
                    filename = os.path.basename(member.filename)
                    target_path = os.path.join(work_dir, filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
            
            # 查找长音频文件
            long_audio_file = Path(work_dir) / f"{package_name}.pcm"
            
            if not long_audio_file.exists():
                pcm_files = list(Path(work_dir).glob("*.pcm"))
                if pcm_files:
                    long_audio_file = pcm_files[0]
                    logger.warning(f"使用第一个PCM文件: {long_audio_file.name}")
                else:
                    raise ValueError("未找到PCM音频文件")
            
            # 加载长音频
            long_audio_data = self.load_pcm_audio(str(long_audio_file))
            logger.info(f"长音频长度: {len(long_audio_data)} samples")
            
            # 使用VAD进行切分
            segments = self.vad_segmentation(long_audio_data)
            
            # 处理VAD切分的音频段
            for segment in segments:
                # 提取音频片段
                audio_segment = self.extract_audio_segment(
                    long_audio_data, 
                    segment['start_time'], 
                    segment['end_time']
                )
                
                # 保存切分后的音频（四位数编号，从0001开始）
                segment_number = segment['segment_id'] + 1  # 从1开始
                segment_filename = f"{package_name}_{segment_number:04d}.wav"
                segment_path = os.path.join(work_dir, segment_filename)
                sf.write(segment_path, audio_segment, 16000)
                
                # 转写切分后的音频
                transcription = self.transcribe_audio(segment_path, "Chinese")
                
                # 计算音频时长
                duration = self.get_audio_duration(audio_segment)
                
                # 记录结果
                result_entry = {
                    'audio_name': segment_filename,
                    'transcription': transcription,
                    'audio_type': 'vad_segment',
                    'duration': duration,  # 添加时长信息（秒）
                    'audio_path': segment_path,  # 添加音频路径用于后续删除
                    'segment_info': {
                        'segment_id': segment['segment_id'],
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'duration': segment['duration']
                    }
                }
                transcription_results.append(result_entry)
                
                self.stats['processed_audios'] += 1
            
            # 处理原有的短音频文件
            short_audio_files = list(Path(work_dir).glob("*.pcm"))
            short_audio_files = [f for f in short_audio_files if f.name != f"{package_name}.pcm"]
            
            for short_audio_file in short_audio_files:
                sid = short_audio_file.stem
                
                # 加载短音频
                short_audio_data = self.load_pcm_audio(str(short_audio_file))
                
                # 保存为WAV格式
                short_wav_path = os.path.join(work_dir, f"{sid}.wav")
                sf.write(short_wav_path, short_audio_data, 16000)
                
                # 转写短音频
                transcription = self.transcribe_audio(short_wav_path, "Chinese")
                
                # 计算短音频时长
                short_duration = self.get_audio_duration(short_audio_data)
                
                # 记录结果（添加时长信息）
                result_entry = {
                    'audio_name': f"{sid}.wav",
                    'transcription': transcription,
                    'audio_type': 'short_audio',
                    'duration': short_duration,  # 添加时长信息（秒）
                    'audio_path': short_wav_path  # 添加音频路径用于后续删除
                }
                transcription_results.append(result_entry)
                
                self.stats['processed_audios'] += 1
            
            # 应用筛选器（如果启用）
            if self.audio_filter and transcription_results:
                original_count = len(transcription_results)
                transcription_results = self.audio_filter.filter_results(transcription_results)
                filtered_count = original_count - len(transcription_results)
                self.stats['filtered_results'] += filtered_count
                self.stats['final_results'] += len(transcription_results)
                logger.info(f"筛选完成: 原始{original_count}条 -> 最终{len(transcription_results)}条")
            
            # 清理解压的PCM文件
            self.cleanup_pcm_files(work_dir)
            
            # 保存当前文件夹的结果到jsonl
            if transcription_results:  # 只有当有结果时才保存
                self.save_folder_results_to_jsonl(transcription_results, work_dir, package_name)
            
            logger.info(f"包 {package_name} 处理完成，共处理 {len(transcription_results)} 个音频文件")
            
        except Exception as e:
            logger.error(f"处理包失败 {package_path}: {e}")
            
        return transcription_results, folder_path
    
    def cleanup_pcm_files(self, work_dir: str):
        """
        清理工作目录中的PCM文件
        
        Args:
            work_dir: 工作目录路径
        """
        try:
            pcm_files = list(Path(work_dir).glob("*.pcm"))
            deleted_count = 0
            
            for pcm_file in pcm_files:
                try:
                    pcm_file.unlink()
                    deleted_count += 1
                    logger.debug(f"已删除PCM文件: {pcm_file}")
                except Exception as e:
                    logger.warning(f"删除PCM文件失败 {pcm_file}: {e}")
            
            if deleted_count > 0:
                logger.info(f"清理完成，共删除 {deleted_count} 个PCM文件")
                
        except Exception as e:
            logger.error(f"清理PCM文件时出错: {e}")
    
    def save_results_to_excel(self, results: List[Dict], output_path: str):
        """
        将转写结果保存到Excel文件
        
        Args:
            results: 转写结果列表
            output_path: 输出Excel文件路径
        """
        try:
            # 准备数据
            excel_data = []
            for result in results:
                entry = {
                    '音频名称': result['audio_name'],
                    '转写结果': result['transcription'],
                    '音频类型': result['audio_type'],
                    '时长(秒)': result.get('duration', 0)  # 添加时长列
                }
                
                # 如果是VAD切分的音频，添加时间段信息
                if result['audio_type'] == 'vad_segment' and 'segment_info' in result:
                    segment_info = result['segment_info']
                    entry['开始时间(秒)'] = segment_info['start_time']/1000.0
                    entry['结束时间(秒)'] = segment_info['end_time']/1000.0
                    entry['持续时间(秒)'] = segment_info['duration']/1000.0
                
                excel_data.append(entry)
            
            # 创建DataFrame并保存
            df = pd.DataFrame(excel_data)
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            logger.info(f"转写结果已保存到Excel文件: {output_path}")
            logger.info(f"共保存 {len(excel_data)} 条记录")
            
        except Exception as e:
            logger.error(f"保存Excel文件失败: {e}")
    
    def batch_process(self, 
                     data_directory: str, 
                     output_directory: str,
                     show_progress: bool = True,
                     remove_empty_folders: bool = True) -> List[Dict]:
        """
        批量处理所有音频包
        
        Args:
            data_directory: 数据目录
            output_directory: 输出目录
            show_progress: 是否显示进度
            remove_empty_folders: 是否删除空文件夹
            
        Returns:
            List[Dict]: 所有转写结果
        """
        zip_files = list(Path(data_directory).glob("*.zip"))
        logger.info(f"发现 {len(zip_files)} 个压缩包")
        
        self.stats['total_packages'] = len(zip_files)
        all_results = []
        processed_folders = []  # 记录处理过的文件夹路径
        
        from tqdm import tqdm
        iterator = tqdm(zip_files, desc="处理音频包") if show_progress else zip_files
        
        for zip_file in iterator:
            logger.info(f"处理: {zip_file.name}")
            package_results, folder_path = self.process_single_audio_package(str(zip_file), output_directory)
            all_results.extend(package_results)
            processed_folders.append((folder_path, len(package_results)))  # 保存文件夹路径和结果数量
            
            if show_progress:
                processed = self.stats['processed_audios']
                successful = self.stats['successful_transcriptions']
                failed = self.stats['failed_transcriptions']
                iterator.set_postfix({
                    '已处理': processed,
                    '成功': successful,
                    '失败': failed
                })
        
        # 处理空文件夹清理
        if remove_empty_folders:
            logger.info("开始检查并清理空文件夹...")
            for folder_path, result_count in processed_folders:
                if result_count == 0:  # 如果该文件夹没有任何处理结果
                    self.remove_empty_folder(folder_path)
        
        # 生成汇总报告
        self.generate_summary_report(all_results, output_directory)
        
        return all_results
    
    def generate_summary_report(self, results: List[Dict], output_dir: str):
        """
        生成处理汇总报告
        
        Args:
            results: 处理结果
            output_dir: 输出目录
        """
        report = {
            'summary': {
                'total_packages': self.stats['total_packages'],
                'total_processed_audios': self.stats['processed_audios'],
                'successful_transcriptions': self.stats['successful_transcriptions'],
                'failed_transcriptions': self.stats['failed_transcriptions'],
                'vad_segments': self.stats['vad_segments'],
                'empty_folders_removed': self.stats['empty_folders_removed'],
                'filtered_results': self.stats.get('filtered_results', 0),
                'final_results': self.stats.get('final_results', 0),
                'success_rate': (
                    self.stats['successful_transcriptions'] / 
                    (self.stats['successful_transcriptions'] + self.stats['failed_transcriptions']) 
                    if (self.stats['successful_transcriptions'] + self.stats['failed_transcriptions']) > 0 
                    else 0
                )
            },
            'type_statistics': {},
            'sample_results': results[:5]  # 显示前5个结果作为示例
        }
        
        # 统计各类型音频数量
        for result in results:
            audio_type = result['audio_type']
            report['type_statistics'][audio_type] = report['type_statistics'].get(audio_type, 0) + 1
        
        # 保存报告
        report_path = os.path.join(output_dir, "processing_summary.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理汇总报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="简化版音频处理器")
    parser.add_argument("--data_dir", default="./data", help="数据目录")
    parser.add_argument("--output_dir", default="./simple_results", help="输出目录")
    parser.add_argument("--asr_model", default="Qwen/Qwen3-ASR-1.7B", help="ASR模型名称")
    parser.add_argument("--vad_model", default="fsmn-vad", 
                       choices=["fsmn-vad", "fireredvad"], 
                       help="VAD模型名称")
    parser.add_argument("--device", default="cuda:0", help="计算设备")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="数据类型")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度显示")
    parser.add_argument("--keep_empty", action="store_true", help="保留空文件夹（不清除）")
    
    # 筛选相关参数
    parser.add_argument("--disable_filter", action="store_true", help="禁用音频筛选功能")
    parser.add_argument("--min_chars", type=int, default=2, help="无标点时的最小字符数")
    parser.add_argument("--similarity", type=float, default=0.6, help="文本相似度阈值")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查FireRedVad可用性
    if args.vad_model.lower() == "fireredvad" and not FIRERED_VAD_AVAILABLE:
        logger.error("FireRedVad 模型不可用，请安装 FireRedASR2S 库")
        logger.error("或者使用默认的 fsmn-vad 模型")
        return
    
    processor = SimpleAudioProcessor(
        asr_model_name=args.asr_model,
        vad_model_name=args.vad_model,
        device=args.device,
        dtype_str=args.dtype,
        enable_filtering=not args.disable_filter,
        min_chars_no_punct=args.min_chars,
        similarity_threshold=args.similarity
    )
    
    # 批量处理
    results = processor.batch_process(
        data_directory=args.data_dir,
        output_directory=args.output_dir,
        show_progress=not args.no_progress,
        remove_empty_folders=not args.keep_empty
    )
    
    # 保存结果到Excel
    excel_path = os.path.join(args.output_dir, "transcription_results.xlsx")
    processor.save_results_to_excel(results, excel_path)
    
    logger.info("批量处理完成!")
    logger.info(f"Excel结果文件: {excel_path}")
    if not args.keep_empty:
        logger.info(f"已清理 {processor.stats['empty_folders_removed']} 个空文件夹")
    
    # 显示筛选统计信息
    if not args.disable_filter:
        logger.info(f"筛选统计: 移除 {processor.stats.get('filtered_results', 0)} 条结果，"
                   f"最终保留 {processor.stats.get('final_results', 0)} 条结果")
    
    # 显示使用的VAD模型信息
    logger.info(f"使用的VAD模型: {processor.vad_wrapper.model_type}")


if __name__ == "__main__":
    main()