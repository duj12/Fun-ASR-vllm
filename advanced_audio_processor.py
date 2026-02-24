#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版音频处理器
包含智能多通道分离、详细对齐分析和质量评估
新增：基于ASR时间戳的长音频切分功能
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf
from qwen_asr import Qwen3ForcedAligner, Qwen3ASRModel

# 配置日志
logging.root.handlers = []  # 清空modelscope修改后的handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_audio_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedAudioProcessor:
    """增强版音频处理器"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
                 asr_model_name: str = "Qwen/Qwen3-ASR-1.7B",
                 device: str = "cuda:0",
                 dtype: torch.dtype = torch.bfloat16):
        """
        初始化处理器
        """
        self.device = device
        self.dtype = dtype
        
        logger.info(f"正在加载强制对齐模型: {model_name}")
        # self.aligner = Qwen3ForcedAligner.from_pretrained(
        #     model_name,
        #     dtype=dtype,
        #     device_map=device,
        # )
        
        logger.info(f"正在加载ASR模型: {asr_model_name}")
        # self.asr_model = Qwen3ASRModel.LLM(
        #     model=asr_model_name,
        #     gpu_memory_utilization=0.7,
        #     max_inference_batch_size=128,
        #     max_new_tokens=4096,
        #     forced_aligner=model_name,
        #     forced_aligner_kwargs=dict(
        #         dtype=dtype,
        #         device_map=device,
        #     ),
        # )   # vllm
        self.asr_model  = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map="cuda:0",
                # attn_implementation="flash_attention_2",
            ),
        )
        
        logger.info("模型加载完成")
        
        # 统计信息
        self.stats = {
            'total_packages': 0,
            'successful_alignments': 0,
            'failed_alignments': 0,
            'channel_separation_success': 0,
            'transcription_success': 0,
            'segmentation_success': 0,
            'quality_assessments': []
        }
    
    def advanced_channel_separation(self, 
                                  audio_data: np.ndarray, 
                                  sample_rate: int = 16000) -> Tuple[np.ndarray, Dict]:
        """
        高级多通道分离算法
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            Tuple[np.ndarray, Dict]: (分离后的用户音频, 分析信息)
        """
        analysis_info = {
            'method': 'unknown',
            'confidence': 0.0,
            'channels_detected': 1,
            'user_channel_index': 0
        }
        
        # 检查是否为立体声音频
        if len(audio_data.shape) == 1:
            # 单声道
            analysis_info.update({
                'method': 'mono',
                'channels_detected': 1,
                'confidence': 1.0
            })
            return audio_data, analysis_info
        
        # 分离左右声道
        left_channel = audio_data[::2]
        right_channel = audio_data[1::2]
        
        # 方法1: 能量差异分析
        left_energy = np.mean(left_channel ** 2)
        right_energy = np.mean(right_channel ** 2)
        energy_ratio = max(left_energy, right_energy) / (min(left_energy, right_energy) + 1e-8)
        
        # 方法2: 频谱特征分析
        left_spectrum = np.abs(np.fft.fft(left_channel[:min(10000, len(left_channel))]))
        right_spectrum = np.abs(np.fft.fft(right_channel[:min(10000, len(right_channel))]))
        
        # 计算频谱相似度（用户语音通常有更多高频成分）
        freq_bins = len(left_spectrum) // 2  # 只考虑正频率
        high_freq_start = int(0.3 * freq_bins)  # 30%以上的频率认为是高频
        
        left_high_freq_energy = np.sum(left_spectrum[high_freq_start:freq_bins] ** 2)
        right_high_freq_energy = np.sum(right_spectrum[high_freq_start:freq_bins] ** 2)
        
        # 方法3: 零交叉率分析（语音通常有更高的零交叉率）
        def zero_crossing_rate(signal):
            return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
        
        left_zcr = zero_crossing_rate(left_channel)
        right_zcr = zero_crossing_rate(right_channel)
        
        # 综合决策
        confidence_scores = []
        
        # 能量权重 (40%)
        if energy_ratio > 2.0:
            energy_winner = 0 if left_energy > right_energy else 1
            confidence_scores.append((energy_winner, min(energy_ratio / 5.0, 1.0) * 0.4))
        
        # 高频能量权重 (35%)
        if abs(left_high_freq_energy - right_high_freq_energy) / (left_high_freq_energy + right_high_freq_energy + 1e-8) > 0.3:
            freq_winner = 0 if left_high_freq_energy > right_high_freq_energy else 1
            confidence_scores.append((freq_winner, 0.35))
        
        # 零交叉率权重 (25%)
        if abs(left_zcr - right_zcr) / (left_zcr + right_zcr + 1e-8) > 0.2:
            zcr_winner = 0 if left_zcr > right_zcr else 1
            confidence_scores.append((zcr_winner, 0.25))
        
        # 决策融合
        if confidence_scores:
            # 加权投票
            score_sum = sum(score for _, score in confidence_scores)
            weighted_votes = {}
            for channel, score in confidence_scores:
                weighted_votes[channel] = weighted_votes.get(channel, 0) + score
            
            winner_channel = max(weighted_votes.items(), key=lambda x: x[1])[0]
            final_confidence = weighted_votes[winner_channel] / score_sum if score_sum > 0 else 0.5
        else:
            # 默认选择能量较高的通道
            winner_channel = 0 if left_energy > right_energy else 1
            final_confidence = 0.5
        
        # 选择用户音频通道
        user_audio = left_channel if winner_channel == 0 else right_channel
        
        analysis_info.update({
            'method': 'advanced_separation',
            'channels_detected': 2,
            'user_channel_index': winner_channel,
            'confidence': final_confidence,
            'metrics': {
                'energy_ratio': float(energy_ratio),
                'left_energy': float(left_energy),
                'right_energy': float(right_energy),
                'left_high_freq_energy': float(left_high_freq_energy),
                'right_high_freq_energy': float(right_high_freq_energy),
                'left_zcr': float(left_zcr),
                'right_zcr': float(right_zcr)
            }
        })
        
        if final_confidence > 0.7:
            self.stats['channel_separation_success'] += 1
            logger.info(f"成功分离多通道音频 (置信度: {final_confidence:.2f})")
        
        return user_audio, analysis_info
    
    def quality_assessment(self, 
                          original_audio: np.ndarray,
                          separated_audio: np.ndarray,
                          alignment_result: List) -> Dict:
        """
        音频质量评估
        
        Args:
            original_audio: 原始音频
            separated_audio: 分离后的音频
            alignment_result: 对齐结果
            
        Returns:
            Dict: 质量评估结果
        """
        assessment = {
            'snr': 0.0,
            'length_ratio': 0.0,
            'alignment_quality': 0.0,
            'overall_score': 0.0
        }
        
        # 计算SNR
        noise_power = np.mean((original_audio - separated_audio[:len(original_audio)]) ** 2)
        signal_power = np.mean(separated_audio ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        assessment['snr'] = float(snr)
        
        # 长度比例
        length_ratio = len(separated_audio) / len(original_audio) if len(original_audio) > 0 else 0
        assessment['length_ratio'] = float(length_ratio)
        
        # 对齐质量评估
        if alignment_result:
            # 计算对齐的连续性和完整性
            word_count = len(alignment_result)
            total_duration = sum(item.end_time - item.start_time for item in alignment_result)
            avg_word_duration = total_duration / word_count if word_count > 0 else 0
            
            # 简单的质量评分 (0-1)
            alignment_quality = min(avg_word_duration / 0.5, 1.0)  # 平均词长不超过0.5秒认为较好
            assessment['alignment_quality'] = float(alignment_quality)
        else:
            assessment['alignment_quality'] = 0.0
        
        # 综合评分
        overall_score = (
            min(snr / 20.0, 1.0) * 0.4 +  # SNR权重40%
            min(abs(length_ratio - 1.0), 1.0) * 0.3 +  # 长度匹配权重30%
            assessment['alignment_quality'] * 0.3  # 对齐质量权重30%
        )
        assessment['overall_score'] = float(overall_score)
        
        self.stats['quality_assessments'].append(assessment)
        
        return assessment
    
    def transcribe_audio_with_timestamps(self, audio_path: str, language: str = "Chinese") -> List:
        """
        使用ASR模型对音频进行转写并生成时间戳
        
        Args:
            audio_path: 音频文件路径
            language: 语言类型
            
        Returns:
            List: 包含时间戳信息的转写结果
        """
        try:
            logger.info(f"正在转写音频: {audio_path}")
            results = self.asr_model.transcribe(
                audio=[audio_path],
                language=[language],
                return_time_stamps=True,
            )
            
            if results and len(results) > 0:
                transcription_result = results[0]
                logger.info(f"转写完成: {transcription_result.text}")
                return transcription_result.time_stamps
            else:
                logger.warning(f"转写结果为空: {audio_path}")
                return []
                
        except Exception as e:
            logger.error(f"ASR转写失败 {audio_path}: {e}")
            return []

    def segment_long_audio_by_timestamps(self, 
                                       long_audio_path: str,
                                       short_audio_segments: List[Dict],
                                       output_dir: str,
                                       package_name: str) -> List[Dict]:
        """
        根据短音频的时间戳信息切分长音频
        
        Args:
            long_audio_path: 长音频文件路径
            short_audio_segments: 短音频段信息列表，每个包含{'sid': str, 'timestamps': List}
            output_dir: 输出目录
            package_name: 包名前缀
            
        Returns:
            List[Dict]: 切分结果信息
        """
        segmentation_results = []
        
        try:
            # 加载长音频
            long_audio_data, sample_rate = sf.read(long_audio_path)
            if len(long_audio_data.shape) > 1:
                long_audio_data = long_audio_data[:, 0]  # 取单声道
            
            logger.info(f"长音频采样率: {sample_rate}, 长度: {len(long_audio_data)} samples")
            
            # 获取长音频的整体时间戳
            long_timestamps = self.transcribe_audio_with_timestamps(long_audio_path, "Chinese")
            
            if not long_timestamps:
                logger.error("长音频转写失败，无法进行切分")
                return segmentation_results
            
            logger.info(f"长音频转写结果: {[item.text for item in long_timestamps]}")
            
            # 为每个短音频段找到对应的时间范围
            for segment_info in short_audio_segments:
                sid = segment_info['sid']
                short_timestamps = segment_info['timestamps']
                
                if not short_timestamps:
                    logger.warning(f"短音频 {sid} 无时间戳信息，跳过")
                    continue
                
                # 根据短音频的文字内容在长音频中定位
                segment_words = [item.text for item in short_timestamps]
                segment_text = ''.join(segment_words)
                logger.info(f"处理短音频段 {sid}: {segment_text}")
                
                # 在长音频时间戳中寻找匹配的词序列
                matched_ranges = self.find_word_sequence_in_timestamps(
                    segment_words, long_timestamps
                )
                
                if matched_ranges:
                    # 合并时间范围
                    start_time = min(range_info['start_time'] for range_info in matched_ranges)
                    end_time = max(range_info['end_time'] for range_info in matched_ranges)
                    
                    # 转换为样本点
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    
                    # 边界检查
                    start_sample = max(0, start_sample)
                    end_sample = min(len(long_audio_data), end_sample)
                    
                    # 提取音频片段
                    audio_segment = long_audio_data[start_sample:end_sample]
                    
                    # 保存切分后的音频
                    output_filename = f"{sid}_orig.wav"
                    output_path = os.path.join(output_dir, output_filename)
                    sf.write(output_path, audio_segment, sample_rate)
                    
                    segmentation_result = {
                        'sid': sid,
                        'original_sid': sid,
                        'output_file': output_filename,
                        'output_path': output_path,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'duration': end_time - start_time,
                        'text': segment_text,
                        'matched_words': len(matched_ranges),
                        'status': 'success'
                    }
                    
                    segmentation_results.append(segmentation_result)
                    self.stats['segmentation_success'] += 1
                    logger.info(f"成功切分 {sid}: {start_time:.2f}s - {end_time:.2f}s")
                else:
                    logger.warning(f"未能在长音频中找到匹配的词序列: {segment_text}")
                    segmentation_results.append({
                        'sid': sid,
                        'status': 'failed',
                        'reason': 'word_sequence_not_found',
                        'text': segment_text
                    })
            
        except Exception as e:
            logger.error(f"音频切分过程出错: {e}")
            
        return segmentation_results

    def find_word_sequence_in_timestamps(self, 
                                       target_words: List[str], 
                                       timestamp_list: List) -> List[Dict]:
        """
        在时间戳列表中寻找目标词序列
        
        Args:
            target_words: 目标词列表
            timestamp_list: 时间戳对象列表
            
        Returns:
            List[Dict]: 匹配的时间范围信息
        """
        matched_ranges = []
        timestamp_words = [item.text for item in timestamp_list]
        
        # 简单的滑动窗口匹配
        target_len = len(target_words)
        timestamp_len = len(timestamp_words)
        
        for i in range(timestamp_len - target_len + 1):
            # 检查当前窗口是否匹配
            window_matches = True
            for j in range(target_len):
                if timestamp_words[i + j] != target_words[j]:
                    window_matches = False
                    break
            
            if window_matches:
                # 找到匹配，记录时间范围
                start_idx = i
                end_idx = i + target_len - 1
                
                range_info = {
                    'start_time': timestamp_list[start_idx].start_time,
                    'end_time': timestamp_list[end_idx].end_time,
                    'words': target_words,
                    'start_index': start_idx,
                    'end_index': end_idx
                }
                matched_ranges.append(range_info)
                
                # 如果只需要第一个匹配，可以在这里break
                # break
        
        return matched_ranges

    def process_audio_package(self, 
                            package_path: str, 
                            output_base_dir: str) -> Dict:
        """
        处理单个音频包（新增ASR转写和音频切分功能）
        """
        package_name = Path(package_path).stem
        work_dir = os.path.join(output_base_dir, package_name)
        os.makedirs(work_dir, exist_ok=True)
        
        result = {
            'package_name': package_name,
            'status': 'processing',
            'separation_analysis': None,
            'transcriptions': {
                'long_audio': None,
                'short_audios': []
            },
            'segmentations': [],
            'alignments': [],
            'quality_scores': [],
            'error': None
        }
        
        try:
            # 解压文件到指定目录（不解压到新的子文件夹）
            import zipfile
            with zipfile.ZipFile(package_path, 'r') as zip_ref:
                # 直接解压到工作目录，不创建额外的文件夹层
                for member in zip_ref.infolist():
                    # 处理文件名编码问题
                    filename = member.filename
                    if filename.endswith('/'):  # 跳过目录
                        continue
                    
                    # 构建目标路径
                    filename = os.path.basename(filename)  # 去掉文件夹，只保留文件名
                    target_path = os.path.join(work_dir, filename)
                    # 确保目录存在
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # 解压文件
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
            
            # 查找数据文件（现在直接在work_dir中查找）
            asr_file = Path(work_dir) / f"asr.txt"
            
            asr_results = []
            if not asr_file.exists():
                logger.error(f"未找到asr.txt文件:{asr_file}")
            else:
                # 加载ASR结果
                with open(asr_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            asr_results.append(json.loads(line))
            
            # 查找长音频（文件名与包名相同）
            long_audio_file = Path(work_dir) / f"{package_name}.pcm"
            
            if not long_audio_file.exists():
                # 如果没找到，尝试其他可能的PCM文件
                pcm_files = list(Path(work_dir).glob("*.pcm"))
                if pcm_files:
                    long_audio_file = pcm_files[0]  # 使用第一个找到的PCM文件
                    logger.warning(f"未找到同名音频文件，使用第一个PCM文件: {long_audio_file.name}")
                else:
                    raise ValueError("未找到长音频文件")
            
            # 处理长音频
            with open(long_audio_file, 'rb') as f:
                raw_audio = np.frombuffer(f.read(), dtype=np.int16)
                audio_float = raw_audio.astype(np.float32) / 32768.0
            
            # 高级通道分离
            user_audio, separation_info = self.advanced_channel_separation(audio_float)
            result['separation_analysis'] = separation_info
            
            # 保存分离后的音频
            separated_path = os.path.join(work_dir, f"{package_name}.wav")
            sf.write(separated_path, user_audio, 16000)
        
            # 新增：对长音频进行ASR转写
            long_audio_transcription = self.transcribe_audio_with_timestamps(separated_path, "Chinese")
            result['transcriptions']['long_audio'] = {
                'text': ''.join([item.text for item in long_audio_transcription]) if long_audio_transcription else "",
                'timestamps': [
                    {
                        'word': item.text,
                        'start_time': float(item.start_time),
                        'end_time': float(item.end_time)
                    } for item in long_audio_transcription
                ] if long_audio_transcription else []
            }
            
            # 收集短音频段信息
            short_audio_segments = []
            
            # 处理每个短音频片段
            for asr_item in asr_results:
                sid = asr_item['sid']
                text = asr_item['asrText']
                
                if not text.strip():
                    continue
                
                # 查找对应短音频
                short_audio_file = Path(work_dir) / f"{sid}.pcm"
                if not short_audio_file.exists():
                    continue
                
                # 加载短音频
                with open(short_audio_file, 'rb') as f:
                    short_raw = np.frombuffer(f.read(), dtype=np.int16)
                    short_audio = short_raw.astype(np.float32) / 32768.0
                
                # 保存短音频
                short_path = os.path.join(work_dir, f"{sid}.wav")
                sf.write(short_path, short_audio, 16000)
                
                # 对短音频进行ASR转写
                short_transcription = self.transcribe_audio_with_timestamps(short_path, "Chinese")
                
                short_segment_info = {
                    'sid': sid,
                    'text': ''.join([item.text for item in short_transcription]) if short_transcription else text,
                    'timestamps': [
                        {
                            'word': item.text,
                            'start_time': float(item.start_time),
                            'end_time': float(item.end_time)
                        } for item in short_transcription
                    ] if short_transcription else [],
                    'original_text': text
                }
                
                result['transcriptions']['short_audios'].append(short_segment_info)
                short_audio_segments.append({
                    'sid': sid,
                    'timestamps': short_transcription,
                    'text': short_segment_info['text']
                })
            
            
            # 新增：基于时间戳切分长音频
            segmentation_results = self.segment_long_audio_by_timestamps(
                separated_path, short_audio_segments, work_dir, package_name
            )
            result['segmentations'] = segmentation_results
            
            result['status'] = 'completed'
            self.stats['transcription_success'] += 1
            
        except Exception as e:
            logger.error(f"处理失败 {package_path}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def batch_process(self, 
                     data_directory: str, 
                     output_directory: str,
                     show_progress: bool = True) -> List[Dict]:
        """
        批量处理所有音频包
        """
        zip_files = list(Path(data_directory).glob("*.zip"))
        logger.info(f"发现 {len(zip_files)} 个压缩包")
        
        self.stats['total_packages'] = len(zip_files)
        results = []
        
        from tqdm import tqdm
        iterator = tqdm(zip_files, desc="处理音频包") if show_progress else zip_files
        
        for zip_file in iterator:
            logger.info(f"处理: {zip_file.name}")
            result = self.process_audio_package(str(zip_file), output_directory)
            results.append(result)
            
            if show_progress:
                successful = self.stats['successful_alignments']
                failed = self.stats['failed_alignments']
                iterator.set_postfix({
                    '成功': successful,
                    '失败': failed,
                    '成功率': f"{successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "0%"
                })
        
        # 生成报告
        self.generate_report(results, output_directory)
        return results
    
    def generate_report(self, results: List[Dict], output_dir: str):
        """生成处理报告（更新统计信息）"""
        report = {
            'summary': {
                'total_packages': self.stats['total_packages'],
                'successful_alignments': self.stats['successful_alignments'],
                'failed_alignments': self.stats['failed_alignments'],
                'channel_separation_success': self.stats['channel_separation_success'],
                'transcription_success': self.stats['transcription_success'],
                'segmentation_success': self.stats['segmentation_success'],
                'success_rate': (
                    self.stats['successful_alignments'] / 
                    (self.stats['successful_alignments'] + self.stats['failed_alignments']) 
                    if (self.stats['successful_alignments'] + self.stats['failed_alignments']) > 0 
                    else 0
                )
            },
            'quality_statistics': {},
            'detailed_results': results
        }
        
        # 计算质量统计
        if self.stats['quality_assessments']:
            quality_metrics = ['snr', 'length_ratio', 'alignment_quality', 'overall_score']
            for metric in quality_metrics:
                values = [qa[metric] for qa in self.stats['quality_assessments']]
                report['quality_statistics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # 保存报告
        report_path = os.path.join(output_dir, "detailed_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="增强版音频处理器")
    parser.add_argument("--data_dir", default="./data", help="数据目录")
    parser.add_argument("--output_dir", default="./enhanced_results", help="输出目录")
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B", help="模型名称")
    parser.add_argument("--device", default="cuda:0", help="计算设备")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度显示")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    processor = AdvancedAudioProcessor(
        model_name=args.model,
        device=args.device
    )
    
    results = processor.batch_process(
        data_directory=args.data_dir,
        output_directory=args.output_dir,
        show_progress=not args.no_progress
    )
    
    logger.info("批量处理完成!")


if __name__ == "__main__":
    main()