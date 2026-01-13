from funasr import AutoModel
import argparse
import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import librosa
import numpy as np
import kaldialign
import unicodedata
import re
import os
import logging
from typing import Iterable, Tuple, List, TextIO, Dict
from collections import defaultdict
from funasr.utils.load_utils import extract_fbank
from torch.nn.utils.rnn import pad_sequence
from tn.chinese.normalizer import Normalizer as ZhNormalizer

# 分布式相关函数
def init_process(rank, world_size, backend='nccl'):
    """初始化分布式进程"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()

# 自定义数据集类
class CustomDataset(Dataset):
    """自定义数据集类，支持wav.scp和text格式"""
    def __init__(self, wav_scp_path, text_path, sr=16000):
        """
        Args:
            wav_scp_path: wav.scp文件路径，格式: utt_id /path/to/audio.wav
            text_path: text文件路径，格式: utt_id 文本内容
            sr: 目标采样率
        """
        self.sr = sr
        self.data = []
        
        # 读取wav.scp文件
        wav_mapping = {}
        with open(wav_scp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, wav_path = parts
                    wav_mapping[utt_id] = wav_path
        
        # 读取text文件
        text_mapping = {}
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    text_mapping[utt_id] = text
        
        # 确保两个文件中的utt_id一致
        common_utt_ids = set(wav_mapping.keys()) & set(text_mapping.keys())
        
        # 构建数据列表
        for utt_id in common_utt_ids:
            self.data.append({
                'utt_id': utt_id,
                'wav_path': wav_mapping[utt_id],
                'text': text_mapping[utt_id]
            })
        
        print(f"Loaded {len(self.data)} samples from custom dataset")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        utt_id = item['utt_id']
        wav_path = item['wav_path']
        text = item['text']
        
        # 加载音频
        try:
            waveform, sample_rate = librosa.load(wav_path, sr=self.sr, mono=True)
            waveform = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # 返回一个空的音频作为占位符
            waveform = torch.zeros((1, self.sr))
            sample_rate = self.sr
        
        # 确保音频是单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 确保音频是float32
        waveform = waveform.float()
        
        return {
            'utt_id': utt_id,
            'audio': {
                'array': waveform.squeeze(0).numpy(),  # 转换为numpy数组
                'sampling_rate': sample_rate
            },
            'text': text
        }


def store_transcripts(
    filename: str, texts: Iterable[Tuple[str, str, str]]
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
) -> float:
    """Write statistics based on predicted results and reference transcripts.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])

    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    if ref_len > 0:
        tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)
    else:
        tot_err_rate = "0.00"

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    if ref_len > 0:
        return float(tot_errs) / float(ref_len) * 100.0
    else:
        return 0.0

def get_args():
    parser = argparse.ArgumentParser(description="FunASR Inference with Custom Dataset")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        help="Model directory"
    )
    parser.add_argument(
        "--wav_scp", 
        type=str,
        required=True,
        help="Path to wav.scp file"
    )
    parser.add_argument(
        "--text", 
        type=str,
        required=True,
        help="Path to text file"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="Device to use for inference (for single GPU mode)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (Note: FunASRNano may only support 1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log_new_pipeline_encoder_trt_opt16_vllm_batch_size_16",
        help="Directory to save the results and stats"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="hypos.txt", 
        help="Output file for transcripts"
    )
    parser.add_argument(
        "--stats_file", 
        type=str, 
        default="wer.txt", 
        help="Output file for error statistics"
    )
    parser.add_argument(
        "--result_text", 
        type=str, 
        default="text_tn", 
        help="Output file for normalized transcriptions (utt_id transcription)"
    )
    parser.add_argument(
        "--itn", 
        type=int, 
        default=1,
        help="Enable ITN post-processing if 1"
    )
    parser.add_argument(
        "--vllm_model_dir",
        type=str,
        default=None,
        help="Directory to the vllm model"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed multi-GPU inference"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the process in distributed mode"
    )
    return parser.parse_args()

class DataCollator:
    def __init__(self, ref_column="text"):
        self.ref_column = ref_column

    def __call__(self, batch):
        ids = []
        wavs = []
        texts = []
        target_sr = 16000
        
        for item in batch:
            # 获取utt_id
            utt_id = item.get("utt_id") or item.get("id") or str(item.get("key", "unknown"))
            ids.append(utt_id)
            
            # 获取参考文本
            ref_text = item.get(self.ref_column, "")
            if not ref_text:
                # 后备方案
                if "text" in item:
                    ref_text = item["text"]
                elif "sentence" in item:
                    ref_text = item["sentence"]
            texts.append(ref_text)
            
            audio_info = item["audio"]
            audio = audio_info["array"]
            sr = audio_info["sampling_rate"]
            
            # 确保音频是float32和正确的采样率
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()
            
            if sr != target_sr:
                 resampler = torchaudio.transforms.Resample(sr, target_sr)
                 audio_tensor = resampler(audio_tensor)
            
            wavs.append(audio_tensor)
                 
        return ids, wavs, texts

def normalize_text_alimeeting(text: str) -> str:
    """
    Text normalization similar to M2MeT challenge baseline.
    See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
    """
    text = text.replace('\u00A0', '') # test_hard
    text = text.replace(" ", "")
    text = text.replace("<sil>", "")
    text = text.replace("<%>", "")
    text = text.replace("<->", "")
    text = text.replace("<$>", "")
    text = text.replace("<#>", "")
    text = text.replace("<_>", "")
    text = text.replace("<space>", "")
    text = text.replace("`", "")
    text = text.replace("&", "")
    text = text.replace(",", "")
    if re.search("[a-zA-Z]", text):
        text = text.upper()
    text = text.replace("Ａ", "A")
    text = text.replace("ａ", "A")
    text = text.replace("ｂ", "B")
    text = text.replace("ｃ", "C")
    text = text.replace("ｋ", "K")
    text = text.replace("ｔ", "T")
    text = text.replace("，", "")
    text = text.replace("丶", "")
    text = text.replace("。", "")
    text = text.replace("、", "")
    text = text.replace("？", "")
    return text

def run_inference_on_rank(rank, world_size, args):
    """在单个GPU上运行推理"""
    print(f"Rank {rank} starting on device cuda:{rank}")
    
    # 初始化分布式进程组
    init_process(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    
    # 加载模型到当前GPU
    model, kwargs = AutoModel.build_model(
        model=args.model_dir, 
        # vad_model="fsmn-vad",
        # vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True, 
        device=device
    )

    if args.vllm_model_dir is not None:
        from vllm import LLM, SamplingParams
        vllm = LLM(
            model=args.vllm_model_dir, 
            enable_prompt_embeds=True, 
            gpu_memory_utilization=0.4, 
            dtype="bfloat16",
            tensor_parallel_size=world_size,
            trust_remote_code=True
        )
        sampling_params = SamplingParams(
            top_p=0.001,
            max_tokens=500,
        )
        model.vllm = vllm
        model.vllm_sampling_params = sampling_params

    tokenizer, frontend = kwargs["tokenizer"], kwargs["frontend"]

    if args.itn==1:
        instruction = "语音转写："
    else:
        instruction = "语音转写，不进行文本规整："
    prompt_prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}"
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_prefix_ids = tokenizer.encode(prompt_prefix)
    prompt_suffix_ids = tokenizer.encode(prompt_suffix)
    prompt_prefix_ids = torch.tensor(prompt_prefix_ids, dtype=torch.int64).to(device)
    prompt_suffix_ids = torch.tensor(prompt_suffix_ids, dtype=torch.int64).to(device)

    # [T,D]
    prompt_prefix_embeddings = model.llm.model.get_input_embeddings()(prompt_prefix_ids)
    prompt_suffix_embeddings = model.llm.model.get_input_embeddings()(prompt_suffix_ids)

    # 加载自定义数据集
    dataset = CustomDataset(args.wav_scp, args.text)
    
    # 创建分布式sampler
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    collator = DataCollator(ref_column="text")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        sampler=sampler,
        shuffle=False
    )

    zh_tn_model = ZhNormalizer(
        cache_dir="./cache",
        remove_erhua=False,
        remove_interjections=False,
        remove_puncts=True,
        overwrite_cache=False,
    )
    
    def normalize_text(text):
        # Normalize full-width characters to half-width
        text = unicodedata.normalize("NFKC", text)
        text = normalize_text_alimeeting(text)
        return zh_tn_model.normalize(text)

    results = []
    all_hypotheses = {}  # 存储所有识别结果

    print(f"Rank {rank} starting inference...")
    iterator = tqdm(dataloader, desc=f"Rank {rank}")
    start_time = time.time()

    for batch_ids, batch_wavs, batch_refs in iterator:
        speech, speech_lengths = extract_fbank(
            batch_wavs,
            frontend=frontend,
            is_final=True,
        )

        speech = speech.to(device)
        speech_lengths = speech_lengths.to(device)

        encoder_out, encoder_out_lens = model.audio_encoder(
            speech, speech_lengths
        )
        encoder_out, encoder_out_lens = model.audio_adaptor(
            encoder_out, encoder_out_lens
        )

        input_embeddings_list = []
        for i in range(len(batch_wavs)):
            speech_embedding = encoder_out[i, :encoder_out_lens[i], :]
            input_embedding = torch.cat([prompt_prefix_embeddings, speech_embedding, prompt_suffix_embeddings], dim=0)
            input_embeddings_list.append(input_embedding)

        if hasattr(model, "vllm"):
            outputs = model.vllm.generate([{
                "prompt_embeds": input_embeddings_list[i],
            } for i in range(len(input_embeddings_list))],
                model.vllm_sampling_params,
                use_tqdm=False,
            )
            response = [output.outputs[0].text for output in outputs]
        else:
            input_embeddings = pad_sequence(input_embeddings_list, batch_first=True, padding_value=0.0)
            input_embeddings = input_embeddings.to(torch.bfloat16)
            
            attention_mask = torch.zeros(input_embeddings.shape[:2], dtype=torch.long, device=device)
            for i, embedding in enumerate(input_embeddings_list):
                attention_mask[i, :embedding.size(0)] = 1 
            llm_kwargs = kwargs.get("llm_kwargs", {})
            generated_ids = model.llm.generate(
                inputs_embeds=input_embeddings,
                max_new_tokens=512,
                attention_mask=attention_mask,
                **llm_kwargs,
            )
            response = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

        for cut_id, ref, hyp in zip(batch_ids, batch_refs, response):
            hyp = normalize_text(hyp).upper()
            ref = normalize_text(ref).upper()
            results.append((cut_id, ref, hyp))
            all_hypotheses[cut_id] = hyp

        # 可选：打印当前批次结果
        if rank == 0:  # 只在rank 0打印
            print(f"Batch results: {response[:2]}...")

    end_time = time.time()
    print(f"Rank {rank} inference time: {end_time - start_time} seconds")
    
    # 创建每个rank的专属日志目录
    rank_log_dir = os.path.join(args.log_dir, f"rank_{rank}")
    os.makedirs(rank_log_dir, exist_ok=True)
    
    # 保存推理时间
    with open(os.path.join(rank_log_dir, "inference_time.txt"), "w") as f:
        f.write(f"Inference time: {end_time - start_time} seconds")
    
    # 保存当前rank的结果
    output_path = os.path.join(rank_log_dir, args.output_file)
    stats_path = os.path.join(rank_log_dir, args.stats_file)
    
    print(f"Rank {rank} saving transcripts to {output_path}...")
    store_transcripts(output_path, results)
    
    print(f"Rank {rank} saving error stats to {stats_path}...")
    with open(stats_path, "w") as f:
        write_error_stats(f, "custom_dataset", results)
    
    # 保存识别结果到text_tn文件
    result_text_path = os.path.join(rank_log_dir, args.result_text)
    with open(result_text_path, "w", encoding="utf-8") as f:
        for utt_id, hyp in sorted(all_hypotheses.items()):
            f.write(f"{utt_id} {hyp}\n")
    
    # 同步所有进程
    dist.barrier()
    
    # 在rank 0收集所有结果
    if rank == 0:
        # 收集所有rank的结果
        all_results = []
        all_hypotheses_all = {}
        
        for r in range(world_size):
            rank_dir = os.path.join(args.log_dir, f"rank_{r}")
            rank_text_path = os.path.join(rank_dir, args.result_text)
            
            if os.path.exists(rank_text_path):
                with open(rank_text_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            utt_id, hyp = parts
                            all_hypotheses_all[utt_id] = hyp
        
        # 保存最终合并的text_tn文件
        final_text_path = os.path.join(args.log_dir, args.result_text)
        with open(final_text_path, "w", encoding="utf-8") as f:
            for utt_id in sorted(all_hypotheses_all.keys()):
                f.write(f"{utt_id} {all_hypotheses_all[utt_id]}\n")
        
        print(f"Final results saved to {final_text_path}")
    
    # 清理分布式进程组
    cleanup()
    
    return results

def run_single_gpu_inference(args):
    """单GPU推理"""
    device=args.device    
    # 加载模型
    model, kwargs = AutoModel.build_model(
        model=args.model_dir, 
        # vad_model="fsmn-vad",
        # vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True, 
        device=device
    )

    if args.vllm_model_dir is not None:
        from vllm import LLM, SamplingParams
        vllm = LLM(
            model=args.vllm_model_dir, 
            enable_prompt_embeds=True, 
            gpu_memory_utilization=0.4, 
            dtype="bfloat16",
            trust_remote_code=True
        )
        sampling_params = SamplingParams(
            top_p=0.001,
            max_tokens=500,
        )
        model.vllm = vllm
        model.vllm_sampling_params = sampling_params

    tokenizer, frontend = kwargs["tokenizer"], kwargs["frontend"]

    if args.itn==1:
        instruction = "语音转写："
    else:
        instruction = "语音转写，不进行文本规整："
    prompt_prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}"
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_prefix_ids = tokenizer.encode(prompt_prefix)
    prompt_suffix_ids = tokenizer.encode(prompt_suffix)
    prompt_prefix_ids = torch.tensor(prompt_prefix_ids, dtype=torch.int64).to(device)
    prompt_suffix_ids = torch.tensor(prompt_suffix_ids, dtype=torch.int64).to(device)

    # [T,D]
    prompt_prefix_embeddings = model.llm.model.get_input_embeddings()(prompt_prefix_ids)
    prompt_suffix_embeddings = model.llm.model.get_input_embeddings()(prompt_suffix_ids)

    # 加载自定义数据集
    dataset = CustomDataset(args.wav_scp, args.text)
    
    collator = DataCollator(ref_column="text")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=False
    )

    zh_tn_model = ZhNormalizer(
        cache_dir="./cache",
        remove_erhua=False,
        remove_interjections=False,
        remove_puncts=True,
        overwrite_cache=False,
    )
    
    def normalize_text(text):
        # Normalize full-width characters to half-width
        text = unicodedata.normalize("NFKC", text)
        text = normalize_text_alimeeting(text)
        return zh_tn_model.normalize(text)

    results = []
    all_hypotheses = {}  # 存储所有识别结果

    print("Starting inference...")
    iterator = tqdm(dataloader)
    start_time = time.time()

    for batch_ids, batch_wavs, batch_refs in iterator:
        speech, speech_lengths = extract_fbank(
            batch_wavs,
            frontend=frontend,
            is_final=True,
        )

        speech = speech.to(device)
        speech_lengths = speech_lengths.to(device)

        encoder_out, encoder_out_lens = model.audio_encoder(
            speech, speech_lengths
        )
        encoder_out, encoder_out_lens = model.audio_adaptor(
            encoder_out, encoder_out_lens
        )

        input_embeddings_list = []
        for i in range(len(batch_wavs)):
            speech_embedding = encoder_out[i, :encoder_out_lens[i], :]
            input_embedding = torch.cat([prompt_prefix_embeddings, speech_embedding, prompt_suffix_embeddings], dim=0)
            input_embeddings_list.append(input_embedding)

        if hasattr(model, "vllm"):
            outputs = model.vllm.generate([{
                "prompt_embeds": input_embeddings_list[i],
            } for i in range(len(input_embeddings_list))],
                model.vllm_sampling_params,
                use_tqdm=False,
            )
            response = [output.outputs[0].text for output in outputs]
        else:
            input_embeddings = pad_sequence(input_embeddings_list, batch_first=True, padding_value=0.0)
            input_embeddings = input_embeddings.to(torch.bfloat16)
            
            attention_mask = torch.zeros(input_embeddings.shape[:2], dtype=torch.long, device=device)
            for i, embedding in enumerate(input_embeddings_list):
                attention_mask[i, :embedding.size(0)] = 1 
            llm_kwargs = kwargs.get("llm_kwargs", {})
            generated_ids = model.llm.generate(
                inputs_embeds=input_embeddings,
                max_new_tokens=512,
                attention_mask=attention_mask,
                **llm_kwargs,
            )
            response = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

        for cut_id, ref, hyp in zip(batch_ids, batch_refs, response):
            hyp = normalize_text(hyp).upper()
            ref = normalize_text(ref).upper()
            results.append((cut_id, ref, hyp))
            all_hypotheses[cut_id] = hyp

        # 打印当前批次结果
        print(f"Batch results: {response[:2]}...")

    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 保存推理时间
    with open(os.path.join(args.log_dir, "inference_time.txt"), "w") as f:
        f.write(f"Inference time: {end_time - start_time} seconds")
    
    output_path = os.path.join(args.log_dir, args.output_file)
    stats_path = os.path.join(args.log_dir, args.stats_file)
    
    print(f"Saving transcripts to {output_path}...")
    store_transcripts(output_path, results)
    
    print(f"Saving error stats to {stats_path}...")
    with open(stats_path, "w") as f:
        write_error_stats(f, "custom_dataset", results)
    
    # 保存识别结果到text_tn文件
    result_text_path = os.path.join(args.log_dir, args.result_text)
    with open(result_text_path, "w", encoding="utf-8") as f:
        for utt_id, hyp in sorted(all_hypotheses.items()):
            f.write(f"{utt_id} {hyp}\n")
    
    print(f"Results saved to {result_text_path}")
    print("Done.")

def main():
    args = get_args()
    
    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.distributed:
        # 多卡并行推理
        if args.world_size is None:
            args.world_size = torch.cuda.device_count()
        
        print(f"Starting distributed inference with {args.world_size} GPUs")
        
        # 使用spawn启动多进程
        mp.spawn(
            run_inference_on_rank,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # 单卡推理
        run_single_gpu_inference(args)

if __name__ == "__main__":
    main()