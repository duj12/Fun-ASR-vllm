import argparse
import json
import re
import os
from typing import Dict, List, Tuple
from tqdm import tqdm


def detect_language(text: str) -> str:
    """检测文本语言"""
    if not text:
        return "<|zh|>"  # 默认中文
    
    # 去除所有标点符号
    text_no_punct = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    
    # 中文字符范围：\u4e00-\u9fff
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_no_punct)
    num_chinese_chars = len(chinese_chars)
    
    # 英文单词（连续的英文字母）
    english_words = re.findall(r'\b[a-zA-Z]+\b', text_no_punct)
    num_english_words = len(english_words)
    total = num_chinese_chars + num_english_words

    if total == 0:
        return "<|zh|>"

    # 判断语言, 英文单词数量超过80% 才算英文
    if num_english_words / total > 0.8:
        return "<|en|>"
    else:
        return "<|zh|>"


def count_text_length(text: str, language: str) -> int:
    """计算文本长度"""
    if not text:
        return 0
     # 1. 统计中文字数（匹配Unicode中的中文字符）
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]')
    chinese_chars = chinese_pattern.findall(text)
    chinese_count = len(chinese_chars)
    
    # 2. 统计英文单词（包含字母，可能包含连字符和撇号）
    english_pattern = re.compile(r'\b[a-zA-Z][a-zA-Z\'\-]*\b')
    english_words = english_pattern.findall(text)
    english_count = len(english_words)
    
    # 3. 统计数字（包括阿拉伯数字、罗马数字等）
    # 匹配独立的数字序列，如123, 3.14, -42, 1,000等
    number_pattern = re.compile(r'\b[-+]?\d[\d,.]*\b')
    numbers = number_pattern.findall(text)
    number_count = len(numbers)
    
    # 4. 统计标点符号（中英文标点）
    # 中文标点：。，、；：！？""''（）《》【】「」『』〔〕…—
    # 英文标点：.,;:!?"'()[]{}<>~@#$%^&*_+=|\\/-
    punctuation_pattern = re.compile(r'[。，、；：！？""''（）《》【】「」『』〔〕…—.,;:!?\"\'()\[\]{}<>~@#$%^&*_+=|\\\/\-]')
    punctuation_chars = punctuation_pattern.findall(text)
    punctuation_count = len(punctuation_chars)
    
    # 按照分类统计的单位数（每个单词/汉字/数字序列/标点都算1个单位）
    total_units = chinese_count + english_count + number_count + punctuation_count
    
    return total_units

def read_key_value_file(file_path: str) -> Dict[str, str]:
    """读取key-value格式的文件"""
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                data[key] = value
            else:
                print(f"Warning: Invalid line format: {line}")
    
    return data


def process_files(args):
    """处理所有文件并生成JSONL"""
    
    # 读取各个文件
    print("Reading wav.scp...")
    wav_scp = read_key_value_file(args.wav_scp)
    
    print("Reading wav2dur...")
    wav2dur = read_key_value_file(args.wav2dur)
    
    # 读取文本文件
    text_tn = {}
    text_itn = {}
    
    if args.text_tn:
        print("Reading text_tn...")
        text_tn = read_key_value_file(args.text_tn)
    
    if args.text_itn:
        print("Reading text_itn...")
        text_itn = read_key_value_file(args.text_itn)
    
    # 验证至少有一个文本文件
    if not text_tn and not text_itn:
        raise ValueError("At least one of text_tn or text_itn must be provided")
    
    # 获取共同的key
    all_keys = set(wav_scp.keys()) & set(wav2dur.keys())
    tn_keys = set(text_tn.keys()) if text_tn else set()
    itn_keys = set(text_itn.keys()) if text_itn else set()
    
    # 找到有效的key（在wav_scp和wav2dur中存在，且至少在一个文本文件中存在）
    valid_keys = all_keys & (tn_keys | itn_keys)
    print(f"Found {len(valid_keys)} valid entries")
    
    if not valid_keys:
        raise ValueError("No valid entries found")
    
    # 处理数据
    results = []
    processed_count = 0
    
    print("Processing data...")
    for key in tqdm(sorted(valid_keys), desc="Converting"):
        try:
            # 获取基础数据
            source = wav_scp[key]
            duration_str = wav2dur[key]
            
            # 计算音频长度（秒数*100取整）
            try:
                duration = float(duration_str)
                source_len = int(round(duration * 100))
            except ValueError:
                print(f"Warning: Invalid duration for {key}: {duration_str}")
                source_len = 0
            
            # 处理TN文本（如果存在）
            if key in text_tn:
                target_tn = text_tn[key]
                
                # 计算语言
                if args.text_language:
                    text_language = args.text_language
                else:
                    text_language = detect_language(target_tn)
                
                # 计算文本长度
                target_len = count_text_length(target_tn, text_language)
                
                # 构造TN记录
                json_obj_tn = {
                    "key": key,
                    "text_language": text_language,
                    "emo_target": "<|NEUTRAL|>",
                    "event_target": "<|Speech|>",
                    "with_or_wo_itn": "<|woitn|>",
                    "target": target_tn,
                    "source": source,
                    "target_len": target_len,
                    "source_len": source_len
                }
                
                results.append(json_obj_tn)
                processed_count += 1
            
            # 处理ITN文本（如果存在）
            if key in text_itn:
                target_itn = text_itn[key]
                
                # 计算语言
                if args.text_language:
                    text_language = args.text_language
                else:
                    text_language = detect_language(target_itn)
                
                # 计算文本长度
                target_len = count_text_length(target_itn, text_language)
                
                # 构造ITN记录
                json_obj_itn = {
                    "key": key,
                    "text_language": text_language,
                    "emo_target": "<|NEUTRAL|>",
                    "event_target": "<|Speech|>",
                    "with_or_wo_itn": "<|withitn|>",
                    "target": target_itn,
                    "source": source,
                    "target_len": target_len,
                    "source_len": source_len
                }
                
                results.append(json_obj_itn)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue
    
    # 保存结果
    print(f"Saving {processed_count} entries to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计信息
    tn_count = sum(1 for item in results if item["with_or_wo_itn"] == "<|woitn|>")
    itn_count = sum(1 for item in results if item["with_or_wo_itn"] == "<|withitn|>")
    
    print(f"Conversion completed! Output saved to {args.output}")
    print(f"Statistics:")
    print(f"  - Total entries processed: {processed_count}")
    print(f"  - TN entries: {tn_count}")
    print(f"  - ITN entries: {itn_count}")
    print(f"  - Output file: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Convert SCP files to SenseVoice JSONL format")
    
    parser.add_argument('--wav_scp', required=True, help='Path to wav.scp file')
    parser.add_argument('--text_tn', help='Path to text_tn file (optional)')
    parser.add_argument('--text_itn', help='Path to text_itn file (optional)')
    parser.add_argument('--wav2dur', required=True, help='Path to wav2dur file')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--text_language', choices=['<|zh|>', '<|en|>'], 
                       help='Specify text language (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # 验证输入文件
    required_files = [args.wav_scp, args.wav2dur]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # 至少需要一个文本文件
    if not args.text_tn and not args.text_itn:
        raise ValueError("Either --text_tn or --text_itn must be provided")
    
    process_files(args)


if __name__ == "__main__":
    main()