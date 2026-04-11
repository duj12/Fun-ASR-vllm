"""
将 wav.scp + text_tn / text_itn 转为 Nano 训练用 JSONL（与 scp2svsjsonl 输入风格一致）。
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen

import soundfile as sf
from modelscope import AutoTokenizer
from tqdm import tqdm


# SenseVoice 风格语种标签 -> 人类可读语种名（用于「语音转写成xx」）
LANG_TAG_TO_PROMPT_NAME = {
    "<|zh|>": "中文",
    "<|en|>": "英文",
}


def detect_language(text: str) -> str:
    """从文本判断语种标签（与 scp2svsjsonl 一致）。"""
    if not text:
        return "<|zh|>"

    text_no_punct = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text_no_punct)
    num_chinese_chars = len(chinese_chars)
    english_words = re.findall(r"\b[a-zA-Z]+\b", text_no_punct)
    num_english_words = len(english_words)
    total = num_chinese_chars + num_english_words

    if total == 0:
        return "<|zh|>"

    if num_english_words / total > 0.8:
        return "<|en|>"
    return "<|zh|>"


def read_key_value_file(file_path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return data

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                data[parts[0]] = parts[1]
            else:
                print(f"Warning: Invalid line format: {line}")
    return data


def pick_language_for_key(
    key: str,
    text: str,
    lang_map: Dict[str, str],
    text_language: Optional[str],
) -> str:
    if text_language:
        return text_language
    if key in lang_map and lang_map[key].strip():
        return lang_map[key].strip()
    return detect_language(text)


def lang_tag_to_prompt_language(lang_tag: str) -> str:
    if lang_tag in LANG_TAG_TO_PROMPT_NAME:
        return LANG_TAG_TO_PROMPT_NAME[lang_tag]
    m = re.match(r"^<\|(.+?)\|>$", lang_tag.strip())
    if m:
        return m.group(1)
    return lang_tag


def get_prompt(
    hotwords: List[str],
    language: Optional[str] = None,
    itn: bool = True,
) -> str:
    if len(hotwords) > 0:
        hotwords_str = ", ".join(hotwords)
        prompt = (
            "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
            "**上下文信息：**\n\n\n"
        )
        prompt += f"热词列表：[{hotwords_str}]\n"
    else:
        prompt = ""
    if language is None:
        prompt += "语音转写"
    else:
        prompt += f"语音转写成{language}"
    if not itn:
        prompt += "，不进行文本规整"
    return prompt + "："


def extract_hotwords_entity_stub(text: str) -> List[str]:
    """
    预留：在此处接入实体词抽取（NER、领域词典、规则等）。
    当前返回空列表；开启热词且命中概率时仅影响是否附加「热词列表」段落。
    """
    _ = text
    return []


def maybe_hotwords(
    text: str,
    enable: bool,
    prob: float,
    rng: random.Random,
) -> List[str]:
    if not enable or prob <= 0:
        return []
    if rng.random() >= prob:
        return []
    return extract_hotwords_entity_stub(text)


def get_duration_sec(wav_path: str) -> float:
    if wav_path.startswith("http"):
        response = urlopen(wav_path)
        if response.status != 200:
            raise RuntimeError(f"WAV not found: {wav_path}")
        audio_file = BytesIO(response.read())
        return float(sf.info(audio_file).duration)
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    return float(sf.info(wav_path).duration)


def build_user_content(prompt: str, wav_path: str) -> str:
    return f"{prompt}<|startofspeech|>!{wav_path}<|endofspeech|>"


def choose_explicit_language(key: str, itn: bool, seed: int) -> bool:
    """
    让「语音转写」与「语音转写成xx」稳定各占一半（按 key + itn/tn 独立划分），
    避免因生成顺序导致某一类样本（如 TN）总落在同一侧。
    """
    variant = "itn" if itn else "tn"
    payload = f"{seed}|{key}|{variant}".encode("utf-8", errors="ignore")
    h = hashlib.md5(payload).digest()
    return (h[0] % 2) == 1


def process_one_job(
    tokenizer,
    job: dict,
) -> Tuple[Optional[dict], Optional[str]]:
    key = job["key"]
    wav_path = job["wav_path"]
    text = job["text"]
    prompt = job["prompt"]
    try:
        duration = job["duration_sec"]
        speech_length = int((duration * 1000 - 25) // 10 + 1)
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": build_user_content(prompt, wav_path),
                },
                {"role": "assistant", "content": text},
            ],
            "speech_length": speech_length,
            "text_length": len(tokenizer.tokenize(text)),
        }
        return data, None
    except Exception as e:
        return None, f"{key}: {e}"


def collect_jobs(
    args,
    wav_scp: Dict[str, str],
    wav2dur: Dict[str, str],
    text_tn: Dict[str, str],
    text_itn: Dict[str, str],
    lang_map: Dict[str, str],
) -> List[dict]:
    # 仅要求 wav.scp 与文本对齐；wav2dur 可只覆盖部分 utt，其余读音频头
    all_keys = set(wav_scp.keys())
    tn_keys = set(text_tn.keys()) if text_tn else set()
    itn_keys = set(text_itn.keys()) if text_itn else set()
    valid_keys = sorted(all_keys & (tn_keys | itn_keys))

    if not valid_keys:
        return []

    rng = random.Random(args.hotword_seed)
    jobs: List[dict] = []

    for key in valid_keys:
        wav_path = wav_scp[key]
        if wav2dur and key in wav2dur:
            try:
                duration_sec = float(wav2dur[key])
            except ValueError:
                duration_sec = None
        else:
            duration_sec = None

        def add_record(text: str, itn: bool) -> None:
            nonlocal jobs
            lang_tag = pick_language_for_key(key, text, lang_map, args.text_language)
            lang_name = lang_tag_to_prompt_language(lang_tag)
            # 各占一半：稳定划分，不依赖写入顺序
            use_explicit_language = choose_explicit_language(key, itn, args.prompt_lang_seed)
            language_for_prompt = lang_name if use_explicit_language else None
            hotwords = maybe_hotwords(text, args.enable_hotwords, args.hotword_prob, rng)
            prompt = get_prompt(hotwords, language_for_prompt, itn=itn)
            jobs.append(
                {
                    "key": key,
                    "wav_path": wav_path,
                    "text": text,
                    "prompt": prompt,
                    "duration_sec": duration_sec,
                    "need_duration_fetch": duration_sec is None,
                }
            )

        if key in text_itn:
            add_record(text_itn[key], itn=True)
        if key in text_tn:
            add_record(text_tn[key], itn=False)

    return jobs


def fill_durations(jobs: List[dict], max_workers: int) -> Tuple[List[str], List[dict]]:
    to_fetch = [j for j in jobs if j.get("need_duration_fetch")]
    errors: List[str] = []
    lock = threading.Lock()

    def fetch_one(j: dict) -> None:
        try:
            d = get_duration_sec(j["wav_path"])
            with lock:
                j["duration_sec"] = d
                j["need_duration_fetch"] = False
        except Exception as e:
            with lock:
                errors.append(f"{j['key']}: {e}")

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(tqdm(ex.map(fetch_one, to_fetch), total=len(to_fetch), desc="Reading duration"))

    ok_jobs = [j for j in jobs if not j.get("need_duration_fetch")]
    return errors, ok_jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="wav.scp + text_tn/text_itn -> Nano JSONL（丰富 prompt，与 scp2svsjsonl 输入类似）"
    )
    parser.add_argument("--wav_scp", required=True, help="wav.scp（utt<TAB>path）")
    parser.add_argument("--text_tn", default=None, help="TN 文本；prompt 带「不进行文本规整」")
    parser.add_argument("--text_itn", default=None, help="ITN 文本；默认规整类 prompt")
    parser.add_argument("--wav2dur", default=None, help="可选；utt<TAB>时长(秒)。不设则从音频读时长")
    parser.add_argument("--output", required=True, help="输出 JSONL")
    parser.add_argument("--max_workers", type=int, default=None, help="读时长线程数，默认 CPU 核数")
    parser.add_argument(
        "--text_language",
        choices=["<|zh|>", "<|en|>"],
        default=None,
        help="强制全数据语种标签（覆盖 language 文件与自动检测）",
    )
    parser.add_argument(
        "--language_file",
        default=None,
        help="每句语种标签文件（与 scp2svsjsonl 一致）",
    )
    parser.add_argument(
        "--prompt_lang_seed",
        type=int,
        default=12345,
        help="控制「语音转写」/「语音转写成xx」50/50 划分的随机种子（按 key 稳定划分）",
    )
    parser.add_argument(
        "--enable_hotwords",
        action="store_true",
        help="开启热词逻辑（按概率调用预留的实体抽取）",
    )
    parser.add_argument(
        "--hotword_prob",
        type=float,
        default=0.3,
        help="开启热词时，对每条样本尝试加热词的概率（0~1）",
    )
    parser.add_argument(
        "--hotword_seed",
        type=int,
        default=42,
        help="热词抽样随机种子",
    )

    args = parser.parse_args()
    max_workers = args.max_workers if args.max_workers is not None else (os.cpu_count() or 4)

    wav_dir = os.path.dirname(os.path.abspath(args.wav_scp))
    if args.text_tn is None:
        cand = os.path.join(wav_dir, "text_tn")
        if os.path.isfile(cand):
            args.text_tn = cand
    if args.text_itn is None:
        cand = os.path.join(wav_dir, "text_itn")
        if os.path.isfile(cand):
            args.text_itn = cand

    if not args.text_tn and not args.text_itn:
        raise ValueError("至少需要 text_tn 或 text_itn（或放在 wav.scp 同目录下）")

    print("Reading wav.scp...")
    wav_scp = read_key_value_file(args.wav_scp)
    wav2dur: Dict[str, str] = {}
    if args.wav2dur:
        print("Reading wav2dur...")
        wav2dur = read_key_value_file(args.wav2dur)
    else:
        all_k = set(wav_scp.keys())
        if all_k:
            # 若同目录存在 wav2dur 则自动加载（与 svs 体验一致）
            cand = os.path.join(wav_dir, "wav2dur")
            if os.path.isfile(cand):
                print(f"Auto-loading wav2dur: {cand}")
                wav2dur = read_key_value_file(cand)

    text_tn = read_key_value_file(args.text_tn) if args.text_tn else {}
    text_itn = read_key_value_file(args.text_itn) if args.text_itn else {}

    lang_map: Dict[str, str] = {}
    if args.language_file:
        lang_map = read_key_value_file(args.language_file)

    jobs = collect_jobs(args, wav_scp, wav2dur, text_tn, text_itn, lang_map)
    if not jobs:
        raise ValueError("没有可生成的条目（检查 key 是否在 wav 与文本中对齐）")

    dur_errors, jobs = fill_durations(jobs, max_workers)
    if dur_errors:
        print(f"Warning: {len(dur_errors)} duration errors (skipped)")
        for e in dur_errors[:10]:
            print(f"  - {e}")
        if len(dur_errors) > 10:
            print(f"  ... and {len(dur_errors) - 10} more")

    if not jobs:
        raise RuntimeError("没有成功获取时长的样本，请检查 wav 路径或 wav2dur")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    processed = 0
    failed = 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one_job, tokenizer, j): j for j in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Writing JSONL"):
                data, err = future.result()
                if data is not None:
                    json.dump(data, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    processed += 1
                else:
                    failed += 1
                    if err:
                        print(err)

    print(f"Done. written={processed} failed={failed} -> {args.output}")


if __name__ == "__main__":
    main()
