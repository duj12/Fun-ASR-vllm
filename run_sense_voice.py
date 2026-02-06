import argparse
import os
import re
import tqdm
from multiprocessing import Process
import threading
from concurrent.futures import ThreadPoolExecutor
import math
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import librosa
import numpy as np

# 导入CustomDataset
from infer_kaldidata import CustomDataset, DataCollator

def init_model():
    from funasr import AutoModel
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        # model_revision="v2.0.4",
        disable_pbar=True,
        disable_log=True,
        disable_update=True,
    )
    return model


def process_batch(model, batch_inputs, batch_languages="auto", use_itn=True):
    """批量推理函数"""
    try:
        results = model.generate(
            input=batch_inputs,
            cache={},
            language="auto",
            use_itn=use_itn,
        )
        return results
    except Exception as e:
        print(f"Batch processing error: {e}")
        return [None] * len(batch_inputs)


def save_results_thread(utt_list, results, file_handles, lock):
    """多线程保存结果到对应文件"""
    for i, (utt, res) in enumerate(zip(utt_list, results)):
        if res is None:
            continue
            
        try:
            text = res["text"]
            pattern = r"<\|[^|]+\|>"
            matches = re.findall(pattern, text)
            
            if len(matches) >= 3:
                text_language, emo_target, event_target = matches[:3]
                
                # 使用锁确保线程安全写入
                with lock:
                    file_handles['language'].write(f"{utt}\t{text_language}\n")
                    file_handles['emotion'].write(f"{utt}\t{emo_target}\n")
                    file_handles['event'].write(f"{utt}\t{event_target}\n")
                    
                    # 定期刷新缓冲区
                    if i % 10 == 0:
                        file_handles['language'].flush()
                        file_handles['emotion'].flush()
                        file_handles['event'].flush()
                        
        except Exception as e:
            print(f"Error processing utterance {utt}: {e}")


def mp_process_scp(args, thread_num, gpu_id, start_idx, chunk_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    predictor = init_model()

    # 创建三个输出文件
    result_language = f"{args.mos_res}_language.{thread_num}"
    result_emotion = f"{args.mos_res}_emotion.{thread_num}"
    result_event = f"{args.mos_res}_event.{thread_num}"
    
    print(f"thread id {thread_num}, save results to:")
    print(f"  language: {result_language}")
    print(f"  emotion: {result_emotion}")
    print(f"  event: {result_event}")
    
    # 打开文件句柄
    fout_language = open(result_language, 'w', encoding='utf-8')
    fout_emotion = open(result_emotion, 'w', encoding='utf-8')
    fout_event = open(result_event, 'w', encoding='utf-8')
    
    file_handles = {
        'language': fout_language,
        'emotion': fout_emotion,
        'event': fout_event
    }
    
    # 创建线程锁
    lock = threading.Lock()
    
    # 使用CustomDataset加载数据
    # 首先需要创建临时的wav.scp文件只包含当前线程需要处理的部分
    temp_wav_scp = f"{args.wav_scp}.temp_{thread_num}"
    
    with open(args.wav_scp, 'r', encoding='utf-8') as fin, \
         open(temp_wav_scp, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i in range(start_idx, start_idx + chunk_num):
                fout.write(line)
    
    # 创建自定义数据集
    dataset = CustomDataset(temp_wav_scp, sr=16000)
    
    # 创建数据加载器
    collator = DataCollator(ref_column="text")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=False
    )
    
    total_samples = len(dataset)
    processed_samples = 0
    
    with tqdm.tqdm(dataloader, desc=f"Thread-{thread_num}") as pbar:
        for batch_ids, batch_wavs, batch_texts in dataloader:
            # # 准备批次数据
            # batch_inputs = []
            # batch_languages = []
            
            # # 将torch tensor转换为numpy数组用于SenseVoice
            # for wav_tensor in batch_wavs:
            #     if isinstance(wav_tensor, torch.Tensor):
            #         # 确保是单声道
            #         if wav_tensor.dim() > 1:
            #             wav_tensor = wav_tensor.mean(dim=0)
            #         # 转换为numpy数组
            #         wav_array = wav_tensor.numpy()
            #         batch_inputs.append(wav_array)
            #     else:
            #         batch_inputs.append(wav_tensor)
                
            #     batch_languages.append("auto")
            
            # if not batch_inputs:
            #     continue
            
            # # 批量推理
            # batch_results = process_batch(predictor, batch_inputs, batch_languages, use_itn=True)
            batch_results = process_batch(predictor, batch_wavs)
            
            # 多线程保存结果
            save_results_thread(batch_ids, batch_results, file_handles, lock)
            
            # 更新进度
            processed_samples += len(batch_ids)
            pbar.update(1)
            pbar.set_postfix({"Processed": f"{processed_samples}/{total_samples}"})
    
    # 关闭文件
    fout_language.close()
    fout_emotion.close()
    fout_event.close()
    
    # 清理临时文件
    try:
        os.remove(temp_wav_scp)
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--wav_scp", default='/data/megastore/SHARE/TTS/VoiceClone2/test/test/wav.scp',
                        help='wav.scp contain the wav pathes.')
    parser.add_argument('-o', "--mos_res", default="/data/megastore/SHARE/TTS/VoiceClone2/test/test/text_lang",
                        help='path to the mos result')
    parser.add_argument('-g', "--gpu_ids", default='0', help='gpu device ID')
    parser.add_argument('-n', "--num_thread", type=int, default=1, help='num of jobs')
    parser.add_argument('-b', "--batch_size", type=int, default=4, help='batch size for inference')
    parser.add_argument('-w', "--num_workers", type=int, default=2, help='number of workers for data loading')
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
                f"num threads: {thread_num}, batch_size: {args.batch_size}, "
                f"num_workers: {args.num_workers}.")
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

    # 合并三个文本文件
    print("Merging results...")
    os.system(f"cat {args.mos_res}_language.* | sort > {args.mos_res}_language")
    os.system(f"cat {args.mos_res}_emotion.* | sort > {args.mos_res}_emotion")
    os.system(f"cat {args.mos_res}_event.* | sort > {args.mos_res}_event")
    
    # 清理临时文件
    os.system(f"rm {args.mos_res}_language.* {args.mos_res}_emotion.* {args.mos_res}_event.*  {args.wav_scp}.temp_*")
    
    print("Processing completed!")
    print(f"Results saved to:")
    print(f"  Language: {args.mos_res}_language")
    print(f"  Emotion: {args.mos_res}_emotion")
    print(f"  Event: {args.mos_res}_event")