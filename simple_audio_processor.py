#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版音频处理器
使用fsmnvad进行音频切分，对切分后的音频和原始短音频进行转写
结果保存到Excel文件中
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import List, Dict
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


class SimpleAudioProcessor:
    """简化版音频处理器"""
    
    def __init__(self, 
                 asr_model_name: str = "Qwen/Qwen3-ASR-1.7B",
                 vad_model_name: str = "fsmn-vad",
                 device: str = "cuda:0",
                 dtype_str: str = "bfloat16"):
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
        
        logger.info(f"正在加载VAD模型: {vad_model_name}")
        self.vad_model = AutoModel(model=vad_model_name, model_revision="v2.0.4")
        
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
            'vad_segments': 0
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
    
    def vad_segmentation(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """
        使用fsmnvad进行音频切分
        
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
            
            # 使用VAD模型进行切分
            vad_result = self.vad_model.generate(input=temp_wav_path)
            
            # 清理临时文件
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            
            segments = []
            if vad_result and len(vad_result) > 0:
                for i, segment_info in enumerate(vad_result[0]['value']):
                    segments.append({
                        'segment_id': i,
                        'start_time': segment_info[0],
                        'end_time': segment_info[1],
                        'duration': segment_info[1] - segment_info[0]
                    })
            
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
            start_time: 开始时间(ms)
            end_time: 结束时间(ms)
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
    
    def process_single_audio_package(self, 
                                   package_path: str, 
                                   output_base_dir: str) -> List[Dict]:
        """
        处理单个音频包
        
        Args:
            package_path: 压缩包路径
            output_base_dir: 输出基础目录
            
        Returns:
            List[Dict]: 转写结果列表
        """
        package_name = Path(package_path).stem
        work_dir = os.path.join(output_base_dir, package_name)
        os.makedirs(work_dir, exist_ok=True)
        
        transcription_results = []
        
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
                
                # 保存切分后的音频
                segment_filename = f"{package_name}_seg_{segment['segment_id']}.wav"
                segment_path = os.path.join(work_dir, segment_filename)
                sf.write(segment_path, audio_segment, 16000)
                
                # 转写切分后的音频
                transcription = self.transcribe_audio(segment_path, "Chinese")
                
                # 记录结果
                result_entry = {
                    'audio_name': segment_filename,
                    'transcription': transcription,
                    'audio_type': 'vad_segment',
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
                
                # 记录结果
                result_entry = {
                    'audio_name': f"{sid}.wav",
                    'transcription': transcription,
                    'audio_type': 'short_audio'
                }
                transcription_results.append(result_entry)
                
                self.stats['processed_audios'] += 1
            
            # 清理解压的PCM文件
            self.cleanup_pcm_files(work_dir)
            
            logger.info(f"包 {package_name} 处理完成，共处理 {len(transcription_results)} 个音频文件")
            
        except Exception as e:
            logger.error(f"处理包失败 {package_path}: {e}")
            
        return transcription_results
    
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
                    '音频类型': result['audio_type']
                }
                
                # 如果是VAD切分的音频，添加时间段信息
                if result['audio_type'] == 'vad_segment' and 'segment_info' in result:
                    segment_info = result['segment_info']
                    entry['开始时间(秒)'] = segment_info['start_time']
                    entry['结束时间(秒)'] = segment_info['end_time']
                    entry['持续时间(秒)'] = segment_info['duration']
                
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
                     show_progress: bool = True) -> List[Dict]:
        """
        批量处理所有音频包
        
        Args:
            data_directory: 数据目录
            output_directory: 输出目录
            show_progress: 是否显示进度
            
        Returns:
            List[Dict]: 所有转写结果
        """
        zip_files = list(Path(data_directory).glob("*.zip"))
        logger.info(f"发现 {len(zip_files)} 个压缩包")
        
        self.stats['total_packages'] = len(zip_files)
        all_results = []
        
        from tqdm import tqdm
        iterator = tqdm(zip_files, desc="处理音频包") if show_progress else zip_files
        
        for zip_file in iterator:
            logger.info(f"处理: {zip_file.name}")
            package_results = self.process_single_audio_package(str(zip_file), output_directory)
            all_results.extend(package_results)
            
            if show_progress:
                processed = self.stats['processed_audios']
                successful = self.stats['successful_transcriptions']
                failed = self.stats['failed_transcriptions']
                iterator.set_postfix({
                    '已处理': processed,
                    '成功': successful,
                    '失败': failed
                })
        
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
    parser.add_argument("--vad_model", default="fsmn-vad", help="VAD模型名称")
    parser.add_argument("--device", default="cuda:0", help="计算设备")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="数据类型")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度显示")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    processor = SimpleAudioProcessor(
        asr_model_name=args.asr_model,
        vad_model_name=args.vad_model,
        device=args.device,
        dtype_str=args.dtype
    )
    
    # 批量处理
    results = processor.batch_process(
        data_directory=args.data_dir,
        output_directory=args.output_dir,
        show_progress=not args.no_progress
    )
    
    # 保存结果到Excel
    excel_path = os.path.join(args.output_dir, "transcription_results.xlsx")
    processor.save_results_to_excel(results, excel_path)
    
    logger.info("批量处理完成!")
    logger.info(f"Excel结果文件: {excel_path}")


if __name__ == "__main__":
    main()