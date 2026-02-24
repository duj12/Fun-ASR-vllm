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
        根据短音频的时间戳信息切分长音频（使用模糊匹配）
        
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
            
            long_audio_text = ''.join([item.text for item in long_timestamps])
            logger.info(f"长音频转写结果: {long_audio_text}")
            
            # 为每个短音频段找到对应的时间范围
            for segment_info in short_audio_segments:
                sid = segment_info['sid']
                short_timestamps = segment_info['timestamps']
                short_text = segment_info.get('text', '')
                
                if not short_text.strip():
                    logger.warning(f"短音频 {sid} 文本为空，跳过")
                    continue
                
                logger.info(f"处理短音频段 {sid}: {short_text}")
                
                # 使用模糊匹配找到最佳位置
                matches = self.fuzzy_text_matching(
                    short_text, long_audio_text, long_timestamps
                )
                
                if matches:
                    # 选择最佳匹配
                    best_match = matches[0]
                    
                    # 转换为样本点
                    start_sample = int(best_match['start_time'] * sample_rate)
                    end_sample = int(best_match['end_time'] * sample_rate)
                    
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
                        'start_time': best_match['start_time'],
                        'end_time': best_match['end_time'],
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'duration': best_match['end_time'] - best_match['start_time'],
                        'text': short_text,
                        'matched_text': best_match.get('matched_text', ''),
                        'similarity': best_match.get('similarity', 0),
                        'edit_distance': best_match.get('edit_distance', 0),
                        'status': 'success'
                    }
                    
                    segmentation_results.append(segmentation_result)
                    self.stats['segmentation_success'] += 1
                    logger.info(f"成功切分 {sid}: {best_match['start_time']:.2f}s - {best_match['end_time']:.2f}s")
                    logger.info(f"匹配相似度: {best_match.get('similarity', 0):.3f}")
                else:
                    logger.warning(f"未能在长音频中找到匹配的文本: {short_text}")
                    segmentation_results.append({
                        'sid': sid,
                        'status': 'failed',
                        'reason': 'text_not_found',
                        'text': short_text
                    })
            
        except Exception as e:
            logger.error(f"音频切分过程出错: {e}")
            
        return segmentation_results

    def find_word_sequence_in_timestamps(self, 
                                       target_words: List[str], 
                                       timestamp_list: List,
                                       max_edit_distance_ratio: float = 0.3) -> List[Dict]:
        """
        在时间戳列表中寻找目标词序列（使用编辑距离优化匹配）
        
        Args:
            target_words: 目标词列表
            timestamp_list: 时间戳对象列表
            max_edit_distance_ratio: 最大编辑距离比例阈值
            
        Returns:
            List[Dict]: 匹配的时间范围信息
        """
        if not target_words or not timestamp_list:
            return []
        
        matched_ranges = []
        timestamp_words = [item.text for item in timestamp_list]
        
        target_text = ''.join(target_words)
        target_len = len(target_text)
        
        if target_len == 0:
            return []
        
        logger.info(f"寻找匹配文本: '{target_text}' (长度: {target_len})")
        logger.info(f"长音频文本: '{''.join(timestamp_words)}'")
        
        best_matches = []
        
        # 滑动窗口搜索最佳匹配
        window_size = max(1, target_len - 2)  # 最小窗口大小
        max_window_size = min(len(''.join(timestamp_words)), target_len + 10)  # 最大窗口大小
        
        for window_size in range(window_size, max_window_size + 1):
            for start_idx in range(len(timestamp_words)):
                # 构建候选文本段
                candidate_words = []
                candidate_indices = []
                current_length = 0
                
                # 从起始位置收集足够的字符
                for i in range(start_idx, len(timestamp_words)):
                    word = timestamp_words[i]
                    if current_length + len(word) <= window_size:
                        candidate_words.append(word)
                        candidate_indices.append(i)
                        current_length += len(word)
                    else:
                        break
                
                if not candidate_words:
                    continue
                
                candidate_text = ''.join(candidate_words)
                
                # 计算编辑距离
                edit_distance = self.calculate_edit_distance(target_text, candidate_text)
                edit_ratio = edit_distance / max(len(target_text), len(candidate_text))
                
                # 记录较好的匹配
                if edit_ratio <= max_edit_distance_ratio:
                    match_info = {
                        'start_index': candidate_indices[0],
                        'end_index': candidate_indices[-1],
                        'candidate_text': candidate_text,
                        'target_text': target_text,
                        'edit_distance': edit_distance,
                        'edit_ratio': edit_ratio,
                        'score': 1.0 - edit_ratio  # 得分越高越好
                    }
                    best_matches.append(match_info)
                    
                    logger.debug(f"候选匹配: '{candidate_text}' vs '{target_text}', "
                               f"编辑距离: {edit_distance}, 比例: {edit_ratio:.3f}")
        
        # 如果没有找到匹配，尝试更宽松的条件
        if not best_matches and max_edit_distance_ratio < 0.5:
            logger.info("未找到严格匹配，尝试更宽松的匹配条件...")
            return self.find_word_sequence_in_timestamps(
                target_words, timestamp_list, max_edit_distance_ratio=0.5
            )
        
        # 按得分排序，选择最好的匹配
        if best_matches:
            best_matches.sort(key=lambda x: x['score'], reverse=True)
            best_match = best_matches[0]
            
            logger.info(f"最佳匹配: '{best_match['candidate_text']}' vs '{best_match['target_text']}'")
            logger.info(f"编辑距离: {best_match['edit_distance']}, 得分: {best_match['score']:.3f}")
            
            # 构建返回的时间范围信息
            range_info = {
                'start_time': timestamp_list[best_match['start_index']].start_time,
                'end_time': timestamp_list[best_match['end_index']].end_time,
                'words': timestamp_words[best_match['start_index']:best_match['end_index']+1],
                'start_index': best_match['start_index'],
                'end_index': best_match['end_index'],
                'match_score': best_match['score'],
                'edit_distance': best_match['edit_distance'],
                'edit_ratio': best_match['edit_ratio']
            }
            matched_ranges.append(range_info)
            
            # 如果还有其他高质量匹配，也可以考虑
            for match in best_matches[1:]:
                if match['score'] >= best_match['score'] * 0.8:  # 至少80%的得分
                    additional_range = {
                        'start_time': timestamp_list[match['start_index']].start_time,
                        'end_time': timestamp_list[match['end_index']].end_time,
                        'words': timestamp_words[match['start_index']:match['end_index']+1],
                        'start_index': match['start_index'],
                        'end_index': match['end_index'],
                        'match_score': match['score'],
                        'edit_distance': match['edit_distance'],
                        'edit_ratio': match['edit_ratio']
                    }
                    matched_ranges.append(additional_range)
        else:
            logger.warning(f"未找到合适的匹配: 目标文本='{target_text}'")
        
        return matched_ranges

    def calculate_edit_distance(self, s1: str, s2: str) -> int:
        """
        计算两个字符串之间的编辑距离（Levenshtein距离）
        
        Args:
            s1: 第一个字符串
            s2: 第二个字符串
            
        Returns:
            int: 编辑距离
        """
        if len(s1) < len(s2):
            return self.calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # 初始化距离矩阵
        previous_row = list(range(len(s2) + 1))
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def fuzzy_text_matching(self, 
                          target_text: str, 
                          source_text: str,
                          timestamp_list: List) -> List[Dict]:
        """
        模糊文本匹配，用于处理不完全匹配的情况
        
        Args:
            target_text: 目标文本（短音频文本）
            source_text: 源文本（长音频文本）
            timestamp_list: 长音频的时间戳列表
            
        Returns:
            List[Dict]: 匹配的位置信息
        """
        if not target_text or not source_text:
            return []
        
        # 清理文本（去除标点符号和空格）
        def clean_text(text):
            import re
            # 保留中文字符和数字
            cleaned = re.sub(r'[^\u4e00-\u9fff0-9]', '', text)
            return cleaned
        
        clean_target = clean_text(target_text)
        clean_source = clean_text(source_text)
        
        logger.info(f"清理后文本对比:")
        logger.info(f"  目标: '{clean_target}'")
        logger.info(f"  源文本: '{clean_source}'")
        
        if not clean_target or not clean_source:
            return []
        
        # 寻找最长公共子序列的位置
        matches = self.find_common_subsequences(clean_target, clean_source, timestamp_list)
        
        # 如果没有找到足够长的匹配，使用滑动窗口方法
        if not matches or max(match['length'] for match in matches) < len(clean_target) * 0.5:
            matches = self.sliding_window_matching(clean_target, clean_source, timestamp_list)
        
        return matches

    def find_common_subsequences(self, target: str, source: str, timestamp_list: List) -> List[Dict]:
        """
        寻找最长公共子序列及其在源文本中的位置
        """
        matches = []
        
        # 简化的匹配算法：寻找目标文本在源文本中的近似位置
        target_len = len(target)
        source_len = len(source)
        
        # 滑动窗口匹配
        best_positions = []
        window_size = max(1, target_len - 2)
        
        for i in range(source_len - window_size + 1):
            window = source[i:i + window_size]
            similarity = self.calculate_similarity(target, window)
            
            if similarity > 0.6:  # 相似度阈值
                # 找到对应的timestamp位置
                start_pos = self.find_timestamp_position(i, source, timestamp_list)
                end_pos = self.find_timestamp_position(i + window_size, source, timestamp_list)
                
                if start_pos is not None and end_pos is not None:
                    match_info = {
                        'start_time': timestamp_list[start_pos].start_time,
                        'end_time': timestamp_list[end_pos].end_time,
                        'similarity': similarity,
                        'matched_text': window,
                        'target_text': target,
                        'length': window_size
                    }
                    matches.append(match_info)
        
        return matches

    def sliding_window_matching(self, target: str, source: str, timestamp_list: List) -> List[Dict]:
        """
        滑动窗口匹配算法
        """
        matches = []
        target_len = len(target)
        
        # 尝试不同的窗口大小
        for window_size in range(max(1, target_len - 3), min(len(source), target_len + 3) + 1):
            for i in range(len(source) - window_size + 1):
                window = source[i:i + window_size]
                edit_dist = self.calculate_edit_distance(target, window)
                similarity = 1 - (edit_dist / max(len(target), len(window)))
                
                if similarity > 0.5:  # 相似度阈值
                    start_pos = self.find_timestamp_position(i, source, timestamp_list)
                    end_pos = self.find_timestamp_position(i + window_size, source, timestamp_list)
                    
                    if start_pos is not None and end_pos is not None:
                        match_info = {
                            'start_time': timestamp_list[start_pos].start_time,
                            'end_time': timestamp_list[end_pos].end_time,
                            'similarity': similarity,
                            'edit_distance': edit_dist,
                            'matched_text': window,
                            'target_text': target,
                            'length': window_size
                        }
                        matches.append(match_info)
        
        # 按相似度排序
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:3]  # 返回前3个最佳匹配

    def calculate_similarity(self, s1: str, s2: str) -> float:
        """
        计算两个字符串的相似度
        """
        if not s1 or not s2:
            return 0.0
        
        # 使用编辑距离计算相似度
        edit_dist = self.calculate_edit_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        similarity = 1 - (edit_dist / max_len) if max_len > 0 else 0
        
        return similarity

    def find_timestamp_position(self, char_position: int, full_text: str, timestamp_list: List) -> Optional[int]:
        """
        根据字符位置找到对应的时间戳索引
        """
        if not timestamp_list or char_position < 0:
            return None
        
        # 构建字符到时间戳的映射
        char_to_timestamp = {}
        current_pos = 0
        
        for idx, item in enumerate(timestamp_list):
            word = item.text
            for char in word:
                char_to_timestamp[current_pos] = idx
                current_pos += 1
        
        # 找到最接近的位置
        if char_position in char_to_timestamp:
            return char_to_timestamp[char_position]
        elif char_position < len(char_to_timestamp):
            # 找到最近的时间戳
            positions = sorted(char_to_timestamp.keys())
            for pos in positions:
                if pos >= char_position:
                    return char_to_timestamp[pos]
            return char_to_timestamp[positions[-1]]
        else:
            return len(timestamp_list) - 1

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