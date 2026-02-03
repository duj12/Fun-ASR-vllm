#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kaldi文本规范化工具
此脚本用于对Kaldi格式的text文件进行文本规范化处理
可以根据语言自动检测并应用相应的文本规范化规则
"""

import argparse
import sys
import re
import unicodedata
import os
from typing import List, Tuple
from tqdm import tqdm

def remove_angle_bracket_content(text: str) -> str:
    """去除文本中由尖括号<>包围的内容（包括尖括号本身）"""
    if not text:
        return text
    
    # 使用正则表达式匹配并去除尖括号及其内容
    # <[^>]*> 匹配 < 开头，任意非>字符，> 结尾的内容
    cleaned_text = re.sub(r'<[^>]*>', '', text)
    
    # 清理多余的空格（如果有连续空格产生的话）
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def detect_language(text: str) -> str:
    """检测文本语言"""
    if not text:
        return "zh"  # 默认中文
    
    # 去除所有中英文标点符号
    text_no_punct = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

    # 中文字符范围：\u4e00-\u9fff
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_no_punct)
    num_chinese_chars = len(chinese_chars)

    # 英文单词（连续的英文字母） 
    english_words = re.findall(r'\b[a-zA-Z]+\b', text_no_punct)
    num_english_words = len(english_words)

    text = str(text)
    total = len(text)
    if total == 0:
        return 'zh'

    en_count = sum(1 for c in text if ord(c) < 128)
    if en_count / total > 0.8:
        return 'en'
    else:
        return 'zh'


def normalize_text(text: str, language: str = "auto") -> str:
    """根据语言进行文本正则化"""
    if not text:
        return text
    
    # 自动检测语言
    if language == "auto":
        language = detect_language(text)
    
    # 中文正则化
    if language == "zh":
        # Normalize full-width characters to half-width
        text = unicodedata.normalize("NFKC", text)
        
        # 导入中文文本正则化工具
        try:
            from tn.chinese.normalizer import Normalizer as ZhNormalizer
            # 初始化中文正则器
            if not hasattr(normalize_text, 'zh_tn_model'):
                normalize_text.zh_tn_model = ZhNormalizer(
                    cache_dir="./cache",
                    remove_erhua=False,
                    remove_interjections=False,
                    remove_puncts=True,
                    overwrite_cache=False,
                )
            # 正则+去标点
            normalized = normalize_text.zh_tn_model.normalize(text)
            return normalized
        except ImportError:
            print("Warning: tn.chinese.normalizer not found, Chinese normalization will be skipped", file=sys.stderr)
            return text
    
    # 英文正则化
    elif language == "en":
        try:
            from tn.english.normalizer import Normalizer as EnNormalizer
            # 初始化英文正则器
            if not hasattr(normalize_text, 'en_tn_model'):
                normalize_text.en_tn_model = EnNormalizer(
                    cache_dir="./cache",
                    overwrite_cache=False,
                )
            # 正则
            normalized = normalize_text.en_tn_model.normalize(text)        
            # 保留字母、数字、中文、空白、连字符和所有格撇号
            text_no_punct = re.sub(r'[^\w\s\u4e00-\u9fff\-\']', ' ', normalized)            
            # 清理多余的空格（如果有连续空格产生的话）
            cleaned_text = re.sub(r'\s+', ' ', text_no_punct).strip()
            return cleaned_text
        except ImportError:
            print("Warning: tn.english.normalizer not found, English normalization will be skipped", file=sys.stderr)
            return text
    
    # 如果没有找到对应语言的正则器，返回原文本
    return text


def read_kaldi_format(file_path: str) -> List[Tuple[str, str]]:
    """
    读取Kaldi格式的文本文件
    每行格式为: utt_id text_content
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 分割行，确保至少有两个元素
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Warning: Line {line_num} in {file_path} does not have both utt_id and text, skipping...", 
                      file=sys.stderr)
                continue
                
            utt_id, text = parts[0], parts[1]
            data.append((utt_id, text))
    
    return data


def write_kaldi_format(data: List[Tuple[str, str]], output_path: str):
    """
    将数据写入Kaldi格式的文本文件
    每行格式为: utt_id text_content
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for utt_id, text in data:
            f.write(f"{utt_id} {text}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Kaldi文本规范化工具 - 对Kaldi格式的text文件进行文本规范化处理，支持自动语言检测"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="输入的Kaldi格式text文件路径 (每行: utt_id text_content)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="输出的Kaldi格式text文件路径 (每行: utt_id text_content)"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="auto",
        choices=["auto", "zh", "en"],
        help="指定语言类型。'auto'表示自动检测语言，默认为auto"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="显示详细处理信息"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"正在读取文件: {args.input}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在", file=sys.stderr)
        sys.exit(1)
    
    # 读取Kaldi格式的文本文件
    try:
        data = read_kaldi_format(args.input)
    except Exception as e:
        print(f"错误: 无法读取输入文件 {args.input}: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"成功读取 {len(data)} 条记录")
        print("开始规范化处理...")
    
    # 处理每条记录
    normalized_data = []
    # 创建进度条
    progress_iter = tqdm(data, desc="处理进度", unit="条")
    
    for utt_id, text in progress_iter:
        if args.verbose:
            print(f"处理 {utt_id}: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        text_without_brackets = remove_angle_bracket_content(text)
        if args.verbose and text_without_brackets != text:
            print(f"  -> 去除尖括号后: {text_without_brackets[:50]}{'...' if len(text_without_brackets) > 50 else ''}")
        text = text_without_brackets

        # 根据指定语言或自动检测语言进行文本规范化
        normalized_text = normalize_text(text, language=args.language)
        
        if args.verbose:
            print(f"  -> 规范化结果: {normalized_text[:50]}{'...' if len(normalized_text) > 50 else ''}")
        
        normalized_data.append((utt_id, normalized_text))
    
    # 写入规范化后的结果
    try:
        write_kaldi_format(normalized_data, args.output)
    except Exception as e:
        print(f"错误: 无法写入输出文件 {args.output}: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"已将规范化结果写入: {args.output}")
        print("处理完成!")


if __name__ == "__main__":
    main()