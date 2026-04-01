#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始音频目录构建 Kaldi 风格数据目录（wav.scp / text / wav2dur / utt2spk / spk2utt），
并生成 text_tn、text_itn，可选调用 SenseVoice 得到语种/情绪/事件，可选 train/test 划分，最后生成 JSONL。

示例:
  python prepare_domain_asr_dataset.py ^
    --audio_dir "\\\\192.168.89.59\\share\\Datasets\\ASR\\SpecificDomain\\WaLi_real\\0127-0315" ^
    --out_dir "\\\\192.168.89.59\\share\\SHARE\\TTS\\VoiceClone1\\my_dataset" ^
    --split_train_test

JSONL 默认写入 out_dir：all.jsonl；若划分 train/test 则另有 train.jsonl、test.jsonl。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 与 run_sense_voice / scp2svsjsonl 同目录
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaldi_text_normalizer import normalize_text, remove_angle_bracket_content
from kaldi_text_itn import inverse_normalize_text

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

_AUDIO_EXT_DEFAULT = ("wav", "flac", "mp3", "m4a", "ogg")


def _strip_extension_basename(name: str) -> str:
    """去掉常见音频后缀，得到与 wav 一致的 utt_id（无后缀）。"""
    n = name.strip()
    for ext in _AUDIO_EXT_DEFAULT:
        suf = "." + ext
        if n.lower().endswith(suf):
            return n[: -len(suf)]
    return Path(n).stem


def _audio_duration_sec(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        import librosa

        return float(librosa.get_duration(path=str(path)))


def _read_kaldi_map(path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not path.is_file():
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                m[parts[0]] = parts[1]
    return m


def _write_kaldi_map(path: Path, mapping: Dict[str, str], sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(mapping.keys()) if sort_keys else list(mapping.keys())
    with open(path, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(f"{k}\t{mapping[k]}\n")


def _filter_kaldi_file(src: Path, dst: Path, keep: Set[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            k = line.split(maxsplit=1)[0]
            if k in keep:
                fout.write(line + "\n")


# ---------------------------------------------------------------------------
# 发现音频与标注
# ---------------------------------------------------------------------------

def _glob_audios(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    out: List[Path] = []
    ext_set = {e.lower().lstrip(".") for e in exts}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in ext_set:
            out.append(p)
    return sorted(out)


def _load_excel_map(excel_path: Path) -> Dict[str, str]:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("读取 Excel 需要 pandas：pip install pandas openpyxl") from e
    df = pd.read_excel(excel_path)
    col_audio = "音频名称"
    col_text = "标注后文本"
    if col_audio not in df.columns or col_text not in df.columns:
        raise ValueError(f"Excel 需包含列 {col_audio!r} 与 {col_text!r}，实际列: {list(df.columns)}")
    m: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row[col_audio]
        if pd.isna(raw):
            continue
        text = row[col_text]
        if pd.isna(text):
            continue
        key = _strip_extension_basename(str(raw).strip())
        m[key] = str(text).strip()
    print(f"find {len(m)} utts.")
    return m


def _find_excel_under(root: Path) -> Optional[Path]:
    for pat in ("*.xlsx", "*.xls"):
        found = list(root.rglob(pat))
        if found:
            return sorted(found)[0]
    return None


def _load_txt_annotations(
    audios: List[Path],
    root: Path,
    utt_key_mode: str,
) -> Dict[str, str]:
    """同目录同名 .txt；或根目录下单一 kaldi 格式 text/transcript 文件。"""
    m: Dict[str, str] = {}

    def utt_for(ap: Path) -> str:
        if utt_key_mode == "relpath":
            rel = ap.relative_to(root)
            return str(rel.with_suffix("")).replace("\\", "_").replace("/", "_")
        return ap.stem

    for ap in audios:
        tp = ap.with_suffix(".txt")
        if tp.is_file():
            body = tp.read_text(encoding="utf-8").strip()
            if not body:
                continue
            lines = body.splitlines()
            first = lines[0].strip()
            if "\t" in first and len(lines) == 1:
                parts = first.split(maxsplit=1)
                if len(parts) == 2:
                    m[parts[0]] = parts[1]
                else:
                    m[utt_for(ap)] = body
            elif "\t" in first:
                parts = first.split(maxsplit=1)
                m[parts[0]] = parts[1] if len(parts) == 2 else ""
            else:
                m[utt_for(ap)] = "\n".join(lines)

    # 根目录 kaldi text
    for name in ("text", "transcript.txt", "all_text.txt"):
        p = root / name
        if p.is_file():
            km = _read_kaldi_map(p)
            for k, v in km.items():
                m.setdefault(k, v)
    return m


# ---------------------------------------------------------------------------
# text_tn / text_itn
# ---------------------------------------------------------------------------

_PUNCT_STRIP = re.compile(r"[^\w\s\u4e00-\u9fff]")


def _to_text_tn(raw: str, apply_tn: bool) -> str:
    t = remove_angle_bracket_content(raw)
    t = t.strip()
    if not t:
        return t
    if apply_tn:
        return normalize_text(t)
    t2 = _PUNCT_STRIP.sub("", t)
    return re.sub(r"\s+", " ", t2).strip()


def _to_text_itn(raw: str) -> str:
    t = remove_angle_bracket_content(raw)
    t = t.strip()
    if not t:
        return t
    return inverse_normalize_text(t)


# ---------------------------------------------------------------------------
# 主构建
# ---------------------------------------------------------------------------

def build_kaldi_and_texts(
    audio_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    exts = tuple(x.strip().lower().lstrip(".") for x in args.formats.split(",") if x.strip())
    audios = _glob_audios(audio_dir, exts)
    if not audios:
        raise FileNotFoundError(f"在 {audio_dir} 下未找到音频（扩展名: {exts}）")

    text_map: Dict[str, str] = {}

    if args.excel:
        excel_path = Path(args.excel)
        if not excel_path.is_file():
            raise FileNotFoundError(excel_path)
        text_map = _load_excel_map(excel_path)
    else:
        auto_excel = _find_excel_under(audio_dir)
        print(f"auto_excel={auto_excel}, args.ignore_excel={args.ignore_excel}")
        if auto_excel and not args.ignore_excel:
            print(f"使用 Excel 标注: {auto_excel}")
            text_map = _load_excel_map(auto_excel)
        else:
            text_map = _load_txt_annotations(audios, audio_dir, args.utt_key_mode)

    wav_map: Dict[str, str] = {}
    utt2spk: Dict[str, str] = {}

    for ap in audios:
        if args.utt_key_mode == "relpath":
            utt = str(ap.relative_to(audio_dir).with_suffix("")).replace("\\", "_").replace("/", "_")
        else:
            utt = ap.stem
        spk = ap.parent.name if ap.parent != audio_dir else "unknown"
        wav_map[utt] = str(ap.resolve())
        utt2spk[utt] = spk

    # 与 Excel 对齐时仅保留 basename 模式
    wav_keys = set(wav_map.keys())
    text_keys = set(text_map.keys())
    only_wav = wav_keys - text_keys
    only_txt = text_keys - wav_keys
    if only_wav:
        print(f"警告: {len(only_wav)} 条音频无对应文本（示例: {list(sorted(only_wav))[:5]}）")
    if only_txt:
        print(f"警告: {len(only_txt)} 条文本无对应音频（示例: {list(sorted(only_txt))[:5]}）")

    common = wav_keys & text_keys
    if not common:
        raise RuntimeError("wav 与 text 的 utt_id 无交集，请检查 Excel「音频名称」是否去掉后缀后与音频文件名一致。")

    # 时长
    dur_map: Dict[str, str] = {}
    nj = max(1, args.dur_workers)

    def job(utt: str) -> Tuple[str, float]:
        return utt, _audio_duration_sec(Path(wav_map[utt]))

    with ThreadPoolExecutor(max_workers=nj) as ex:
        futs = [ex.submit(job, u) for u in common]
        for fu in as_completed(futs):
            utt, sec = fu.result()
            dur_map[utt] = f"{sec:.3f}"

    # 时长过滤
    kept: Set[str] = set()
    for utt in common:
        try:
            d = float(dur_map[utt])
        except ValueError:
            continue
        if args.min_duration <= d <= args.max_duration:
            kept.add(utt)
        else:
            print(f"过滤时长外 utterance: {utt} ({d}s)")

    if not kept:
        raise RuntimeError("过滤时长后无剩余 utterance")

    wav_f = {k: wav_map[k] for k in kept}
    text_f = {k: text_map[k] for k in kept}
    utt2spk_f = {k: utt2spk[k] for k in kept}
    dur_f = {k: dur_map[k] for k in kept}

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_kaldi_map(out_dir / "wav.scp", wav_f)
    _write_kaldi_map(out_dir / "text", text_f)
    _write_kaldi_map(out_dir / "wav2dur", dur_f)

    with open(out_dir / "utt2spk", "w", encoding="utf-8") as f:
        for u in sorted(utt2spk_f.keys()):
            f.write(f"{u}\t{utt2spk_f[u]}\n")

    spk2utt: Dict[str, List[str]] = defaultdict(list)
    for u, s in utt2spk_f.items():
        spk2utt[s].append(u)
    with open(out_dir / "spk2utt", "w", encoding="utf-8") as f:
        for s in sorted(spk2utt.keys()):
            f.write(f"{s}\t{' '.join(sorted(spk2utt[s]))}\n")

    text_tn: Dict[str, str] = {}
    text_itn: Dict[str, str] = {}
    for u, raw in text_f.items():
        text_tn[u] = _to_text_tn(raw, args.apply_text_tn)
        text_itn[u] = _to_text_itn(raw)
    _write_kaldi_map(out_dir / "text_tn", text_tn)
    _write_kaldi_map(out_dir / "text_itn", text_itn)

    meta = {
        "n_audio_found": len(audios),
        "n_after_align": len(common),
        "n_after_duration": len(kept),
        "only_wav_no_text": len(only_wav),
        "only_text_no_wav": len(only_txt),
    }
    with open(out_dir / "prepare_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return wav_f, text_f, text_tn, text_itn, dur_f


def run_sense_voice_cli(out_dir: Path, prefix_name: str, args: argparse.Namespace) -> None:
    wav_scp = out_dir / "wav.scp"
    out_prefix = out_dir / prefix_name
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_sense_voice.py"),
        "-i",
        str(wav_scp),
        "-o",
        str(out_prefix),
        "-g",
        args.gpu_ids,
        "-n",
        str(args.sensevoice_threads),
        "-b",
        str(args.sensevoice_batch_size),
        "-w",
        str(args.sensevoice_num_workers),
    ]
    print("运行 SenseVoice:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def split_train_test(
    out_dir: Path,
    test_ratio: float,
    seed: int,
    sensevoice_prefix_name: str,
) -> Tuple[Set[str], Set[str]]:
    wav_scp = out_dir / "wav.scp"
    keys = []
    with open(wav_scp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keys.append(line.split(maxsplit=1)[0])
    n = len(keys)
    if n == 0:
        raise RuntimeError("wav.scp 为空")
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test = min(n - 1, max(0, int(round(n * test_ratio)))) if n > 1 else 0
    test_keys = set(keys[:n_test]) if n_test else set()
    train_keys = set(keys[n_test:])
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"

    for name in (
        "wav.scp",
        "text",
        "text_tn",
        "text_itn",
        "wav2dur",
        "utt2spk",
    ):
        src = out_dir / name
        if not src.is_file():
            continue
        if train_keys:
            _filter_kaldi_file(src, train_dir / name, train_keys)
        if test_keys:
            _filter_kaldi_file(src, test_dir / name, test_keys)

    # spk2utt 由 train/test 的 utt2spk 重建
    def rebuild_spk2utt(udir: Path, keys: Set[str]) -> None:
        if not keys:
            return
        u2s = _read_kaldi_map(udir / "utt2spk")
        spk2utt: Dict[str, List[str]] = defaultdict(list)
        for u, s in u2s.items():
            spk2utt[s].append(u)
        with open(udir / "spk2utt", "w", encoding="utf-8") as f:
            for s in sorted(spk2utt.keys()):
                f.write(f"{s}\t{' '.join(sorted(spk2utt[s]))}\n")

    if train_keys:
        rebuild_spk2utt(train_dir, train_keys)
    if test_keys:
        rebuild_spk2utt(test_dir, test_keys)

    # sense_voice 输出
    for suf in ("_language", "_emotion", "_event"):
        src = out_dir / f"{sensevoice_prefix_name}{suf}"
        if src.is_file():
            if train_keys:
                _filter_kaldi_file(src, train_dir / src.name, train_keys)
            if test_keys:
                _filter_kaldi_file(src, test_dir / src.name, test_keys)

    print(f"划分完成: train={len(train_keys)}, test={len(test_keys)}")
    return train_keys, test_keys


def run_scp2svsjsonl(
    data_dir: Path,
    output_jsonl: Path,
    sensevoice_prefix: Path,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "scp2svsjsonl.py"),
        "--wav_scp",
        str(data_dir / "wav.scp"),
        "--wav2dur",
        str(data_dir / "wav2dur"),
        "--text_tn",
        str(data_dir / "text_tn"),
        "--text_itn",
        str(data_dir / "text_itn"),
        "--output",
        str(output_jsonl),
        "--sensevoice_prefix",
        str(sensevoice_prefix),
    ]
    print("生成 JSONL:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="领域 ASR 数据：Kaldi 目录 + TN/ITN + SenseVoice + JSONL")
    p.add_argument("--audio_dir", required=True, type=Path, help="原始音频根目录")
    p.add_argument("--out_dir", required=True, type=Path, help="输出目录（如 250Hours_zh 结构）")
    p.add_argument("--formats", default="wav,flac,mp3", help="逗号分隔扩展名")
    p.add_argument("--excel", type=Path, default=None, help="指定 Excel（列：音频名称、标注后文本）；不指定则自动搜 .xlsx")
    p.add_argument("--ignore_excel", action="store_true", help="即使目录有 xlsx 也改用 txt 配对")
    p.add_argument(
        "--utt_key_mode",
        choices=("basename", "relpath"),
        default="basename",
        help="utt_id：仅文件名无后缀，或相对 audio_dir 的路径（防重名）",
    )
    p.add_argument("--apply_text_tn", action="store_true", help="text_tn 使用 kaldi_text_normalizer 完整 TN；否则仅去标点")
    p.add_argument("--min_duration", type=float, default=0.1)
    p.add_argument("--max_duration", type=float, default=40.0)
    p.add_argument("--dur_workers", type=int, default=8)

    p.add_argument("--skip_sense_voice", action="store_true", help="不调用 run_sense_voice.py")
    p.add_argument("--sensevoice_prefix_name", default="sense_voice", help="输出前缀：{out_dir}/{name}_language 等")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--sensevoice_threads", type=int, default=1)
    p.add_argument("--sensevoice_batch_size", type=int, default=32)
    p.add_argument("--sensevoice_num_workers", type=int, default=2)

    p.add_argument("--split_train_test", action="store_true")
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)

    p.add_argument("--no_jsonl", action="store_true", help="不生成 JSONL（默认在 out_dir 下生成）")
    p.add_argument(
        "--jsonl_all",
        type=Path,
        default=None,
        help="覆盖默认全量 JSONL 路径（默认：<out_dir>/all.jsonl）",
    )
    p.add_argument("--jsonl_train", type=Path, default=None, help="覆盖默认（默认：<out_dir>/train.jsonl）")
    p.add_argument("--jsonl_test", type=Path, default=None, help="覆盖默认（默认：<out_dir>/test.jsonl）")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio_dir = args.audio_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(audio_dir)

    build_kaldi_and_texts(audio_dir, out_dir, args)

    if not args.skip_sense_voice:
        run_sense_voice_cli(out_dir, args.sensevoice_prefix_name, args)
    else:
        print("已跳过 SenseVoice（--skip_sense_voice）")

    sense_prefix = out_dir / args.sensevoice_prefix_name

    if args.split_train_test:
        split_train_test(out_dir, args.test_ratio, args.split_seed, args.sensevoice_prefix_name)

    if args.no_jsonl:
        print("已跳过 JSONL（--no_jsonl）")
    else:
        jsonl_all_path = args.jsonl_all if args.jsonl_all is not None else out_dir / "all.jsonl"
        run_scp2svsjsonl(out_dir, jsonl_all_path, sense_prefix)

        train_prefix = out_dir / "train" / args.sensevoice_prefix_name
        test_prefix = out_dir / "test" / args.sensevoice_prefix_name

        jsonl_train_path = args.jsonl_train if args.jsonl_train is not None else out_dir / "train.jsonl"
        jsonl_test_path = args.jsonl_test if args.jsonl_test is not None else out_dir / "test.jsonl"

        if args.split_train_test:
            if (out_dir / "train" / "wav.scp").is_file():
                run_scp2svsjsonl(out_dir / "train", jsonl_train_path, train_prefix)
            else:
                print("警告: 未找到 train/wav.scp，跳过 train.jsonl")

            if (out_dir / "test" / "wav.scp").is_file():
                run_scp2svsjsonl(out_dir / "test", jsonl_test_path, test_prefix)
            else:
                print("警告: 未找到 test/wav.scp，跳过 test.jsonl")

    print("完成。输出目录:", out_dir)


if __name__ == "__main__":
    main()
