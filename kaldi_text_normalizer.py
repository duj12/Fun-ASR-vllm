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
from multiprocessing import Process

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
    if en_count / total > 0.2:  # 只要有一点英文，就算是英文。除非很少才视为纯中文
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
                    overwrite_cache=True,
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
                    overwrite_cache=True,
                )
            # 正则
            normalized = normalize_text.en_tn_model.normalize(text)        
            # 保留字母、数字、中文、空白、连字符和所有格撇号，还有/表示音标的分割符
            text_no_punct = re.sub(r'[^\w\s\u4e00-\u9fff\-\'\/]', ' ', normalized)            
            # 清理多余的空格（如果有连续空格产生的话）
            cleaned_text = re.sub(r'\s+', ' ', text_no_punct).strip()
            return cleaned_text
        except ImportError:
            print("Warning: tn.english.normalizer not found, English normalization will be skipped", file=sys.stderr)
            return text
    
    # 如果没有找到对应语言的正则器，返回原文本
    return text


def mp_process_scp(args, thread_num, gpu_id, start_idx, chunk_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = f"{args.mos_res}.{thread_num}"
    print(f"thread id {thread_num}, save result to {result}")
    fout = open(result, 'w', encoding='utf-8')

    with open(args.wav_scp, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(tqdm(fin)):
            if not i in range(start_idx, start_idx + chunk_num):
                continue
            try:
                line = line.strip().split(maxsplit=1)
                if not len(line) == 2:
                    print(f"line: {line} not in kaldi format.")
                    continue
                utt, text = line[0], line[1]

                text_without_brackets = remove_angle_bracket_content(text)
                text = text_without_brackets
                normalized_text = normalize_text(text)

                fout.write(f"{utt}\t{normalized_text}\n")
                fout.flush()
            except Exception as e:
                print(f"Exception: {e}")
                continue

    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--wav_scp", default='/data/megastore/SHARE/TTS/VoiceClone2/test/test/text',
                        help='wav.scp contain the wav pathes.')
    parser.add_argument('-o', "--mos_res", default="/data/megastore/SHARE/TTS/VoiceClone2/test/test/text_punc",
                        help='path to the mos result')
    parser.add_argument('-g', "--gpu_ids", default='0', help='gpu device ID')
    parser.add_argument('-n', "--num_thread", type=int, default=2, help='num of jobs')
    args = parser.parse_args()

    gpus = args.gpu_ids
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpu_list = gpus.split(',')
    gpu_num = len(gpu_list)
    thread_num = int(args.num_thread)
    gpu_list *= thread_num

    wav_scp = args.wav_scp
    output_path = args.mos_res

    f_scp = open(wav_scp)
    total_len = 0
    for line in f_scp:
        total_len += 1

    thread_num = min(thread_num, total_len)
    print(f"Total wavs: {total_len}. gpus: {gpus}, "
                f"num threads: {thread_num}.")
    if total_len >= thread_num:
        chunk_size = int(total_len / thread_num)
        remain_wavs = total_len - chunk_size * thread_num
    else:
        chunk_size = 1
        remain_wavs = 0

    process_list = []
    chunk_begin = 0
    for i in range(thread_num):
        now_chunk_size = chunk_size
        if remain_wavs > 0:
            now_chunk_size = chunk_size + 1
            remain_wavs = remain_wavs - 1
        # process i handle wavs at chunk_begin and size of now_chunk_size
        gpu_id = i % gpu_num
        p = Process(target=mp_process_scp, args=(
            args, i, gpu_list[gpu_id], chunk_begin, now_chunk_size))
        chunk_begin = chunk_begin + now_chunk_size
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    os.system(f"cat {args.mos_res}.* | sort > {args.mos_res}")
    os.system(f"rm {args.mos_res}.* ")