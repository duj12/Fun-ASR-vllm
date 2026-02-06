import argparse
import os
import re
import tqdm
from multiprocessing import Process
import threading
from concurrent.futures import ThreadPoolExecutor
import math

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


def process_batch(model, batch_inputs, batch_languages, use_itn=True):
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
    
    # 读取所有需要处理的行
    lines = []
    with open(args.wav_scp, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            if i in range(start_idx, start_idx + chunk_num):
                lines.append(line.strip())
    
    # 批量处理
    batch_size = args.batch_size
    total_batches = math.ceil(len(lines) / batch_size)
    
    with tqdm.tqdm(total=len(lines), desc=f"Thread-{thread_num}") as pbar:
        for batch_idx in range(total_batches):
            start_pos = batch_idx * batch_size
            end_pos = min((batch_idx + 1) * batch_size, len(lines))
            batch_lines = lines[start_pos:end_pos]
            
            # 准备批次数据
            batch_inputs = []
            batch_languages = []
            batch_utts = []
            
            for line in batch_lines:
                try:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        utt, input_wav = parts[0], parts[1]
                        batch_utts.append(utt)
                        batch_inputs.append(input_wav)
                        batch_languages.append("auto")  # 可以根据需要调整
                except Exception as e:
                    print(f"Error parsing line: {line}, error: {e}")
                    continue
            
            if not batch_inputs:
                continue
                
            # 批量推理
            batch_results = process_batch(predictor, batch_inputs, batch_languages, use_itn=True)
            
            # 多线程保存结果
            save_results_thread(batch_utts, batch_results, file_handles, lock)
            
            # 更新进度
            pbar.update(len(batch_lines))
    
    # 关闭文件
    fout_language.close()
    fout_emotion.close()
    fout_event.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--wav_scp", default='/data/megastore/SHARE/TTS/VoiceClone2/test/test/wav.scp',
                        help='wav.scp contain the wav pathes.')
    parser.add_argument('-o', "--mos_res", default="/data/megastore/SHARE/TTS/VoiceClone2/test/test/text_lang",
                        help='path to the mos result')
    parser.add_argument('-g', "--gpu_ids", default='0', help='gpu device ID')
    parser.add_argument('-n', "--num_thread", type=int, default=1, help='num of jobs')
    parser.add_argument('-b', "--batch_size", type=int, default=4, help='batch size for inference')
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
                f"num threads: {thread_num}, batch_size: {args.batch_size}.")
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
    os.system(f"rm {args.mos_res}_language.* {args.mos_res}_emotion.* {args.mos_res}_event.*")
    
    print("Processing completed!")
    print(f"Results saved to:")
    print(f"  Language: {args.mos_res}_language")
    print(f"  Emotion: {args.mos_res}_emotion")
    print(f"  Event: {args.mos_res}_event")