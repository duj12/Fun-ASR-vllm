#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio concat and split tool

Stage 1: Group by duration -> pad silence -> concat to ~12h audio files
Stage 2: Align recorded audio with original (cross-correlation offset detection)
Stage 3: Split aligned audio by original segment duration, restore text
"""

import os
import argparse
import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
GROUP_CFGS = [
    {"name": "10s", "max_dur": 10.0, "pad_to": 10.0},
    {"name": "20s", "max_dur": 20.0, "pad_to": 20.0},
    {"name": "30s", "max_dur": 30.0, "pad_to": 30.0},
]
HOURS_PER_FILE = 12


def read_kv_file(path: str) -> Dict[str, str]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logger.warning(f"skip bad line: {line}")
                continue
            data[parts[0]] = parts[1]
    return data


def load_audio_mono(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    #audio, sr = sf.read(path, dtype="float32")
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"SR mismatch: {path} is {sr}Hz, expected {target_sr}Hz")
    return audio


def pad_silence(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) >= target_samples:
        return audio[:target_samples]
    return np.pad(audio, (0, target_samples - len(audio)), mode="constant")


def write_text_list(path: str, items: List[Tuple[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for utt_id, text in items:
            f.write(f"{utt_id}\t{text}\n")


def stage1_concat(wav_scp, text_tn_file, text_itn_file, wav2dur_file,
                  output_dir, sample_rate=SAMPLE_RATE):
    wav_dict = read_kv_file(wav_scp)
    tn_dict = read_kv_file(text_tn_file)
    itn_dict = read_kv_file(text_itn_file)
    dur_dict = read_kv_file(wav2dur_file)
    os.makedirs(output_dir, exist_ok=True)

    groups: Dict[str, List[str]] = {cfg["name"]: [] for cfg in GROUP_CFGS}
    for utt_id, dur_str in dur_dict.items():
        dur = float(dur_str)
        if utt_id not in wav_dict:
            logger.warning(f"wav.scp missing {utt_id}, skip")
            continue
        assigned = False
        for cfg in GROUP_CFGS:
            if dur <= cfg["max_dur"]:
                groups[cfg["name"]].append(utt_id)
                assigned = True
                break
        if not assigned:
            logger.info(f"dur {dur:.2f}s > 30s, skip {utt_id}")

    samples_per_file = int(HOURS_PER_FILE * 3600 * sample_rate)

    for cfg in GROUP_CFGS:
        gname = cfg["name"]
        pad_samples = int(cfg["pad_to"] * sample_rate)
        utt_ids = groups[gname]
        if not utt_ids:
            logger.info(f"group {gname}: no data, skip")
            continue
        logger.info(f"group {gname}: {len(utt_ids)} utts")

        buf = np.array([], dtype=np.float32)
        tn_buf: List[Tuple[str, str]] = []
        itn_buf: List[Tuple[str, str]] = []
        fidx = 1

        for i, utt_id in enumerate(utt_ids):
            audio = load_audio_mono(wav_dict[utt_id], sample_rate)
            audio = pad_silence(audio, pad_samples)
            buf = np.concatenate([buf, audio])
            tn_buf.append((utt_id, tn_dict.get(utt_id, "")))
            itn_buf.append((utt_id, itn_dict.get(utt_id, "")))

            if len(buf) >= samples_per_file or i == len(utt_ids) - 1:
                out_name = f"{gname}_{fidx:02d}"
                sf.write(os.path.join(output_dir, f"{out_name}.wav"), buf, sample_rate)
                write_text_list(os.path.join(output_dir, f"{out_name}_tn.txt"), tn_buf)
                write_text_list(os.path.join(output_dir, f"{out_name}_itn.txt"), itn_buf)
                logger.info(
                    f"  {out_name}.wav  dur={len(buf)/sample_rate/3600:.2f}h  n={len(tn_buf)}"
                )
                buf = np.array([], dtype=np.float32)
                tn_buf = []
                itn_buf = []
                fidx += 1


def _find_offset_xcorr(ref, rec, search_range_sec=30.0, sr=SAMPLE_RATE):
    from numpy.fft import fft, ifft

    tpl_len = int(min(10.0, len(ref) / sr) * sr)
    tpl = ref[:tpl_len].astype(np.float64)
    search_len = int(search_range_sec * sr) + tpl_len
    region = rec[:min(search_len, len(rec))].astype(np.float64)
    if len(region) < len(tpl):
        return 0
    n = len(region)
    t = np.zeros(n, dtype=np.float64)
    t[:len(tpl)] = tpl[::-1]
    corr = np.real(ifft(fft(region) * fft(t)))
    return int(np.argmax(corr[:n - len(tpl) + 1]))


def stage2_align(concat_wav, recorded_wav, output_wav,
                 search_range_sec=30.0, sample_rate=SAMPLE_RATE):
    ref = load_audio_mono(concat_wav, sample_rate)
    rec = load_audio_mono(recorded_wav, sample_rate)
    offset = _find_offset_xcorr(ref, rec, search_range_sec, sample_rate)
    logger.info(f"offset: {offset} samples ({offset / sample_rate:.3f}s)")

    aligned = rec[offset:offset + len(ref)]
    if len(aligned) < len(ref):
        aligned = pad_silence(aligned, len(ref))
        logger.warning("recorded audio too short, padded silence at tail")

    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
    sf.write(output_wav, aligned, sample_rate)
    logger.info(f"aligned -> {output_wav}  dur={len(aligned)/sample_rate/3600:.2f}h")


def _read_text_items(path: str) -> List[Tuple[str, str]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            items.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return items


def stage3_split(aligned_wav, concat_tn_txt, concat_itn_txt, output_dir,
                 segment_sec=10.0, sample_rate=SAMPLE_RATE):
    audio = load_audio_mono(aligned_wav, sample_rate)
    seg_samples = int(segment_sec * sample_rate)

    tn_items = _read_text_items(concat_tn_txt)
    itn_items = _read_text_items(concat_itn_txt)
    if len(tn_items) != len(itn_items):
        logger.warning(
            f"text_tn ({len(tn_items)}) and text_itn ({len(itn_items)}) "
            f"have different line counts, using min"
        )
    n_segs = min(len(tn_items), len(itn_items))

    base_name = Path(aligned_wav).stem
    os.makedirs(output_dir, exist_ok=True)
    new_tn: List[Tuple[str, str]] = []
    new_itn: List[Tuple[str, str]] = []

    for idx in range(n_segs):
        start = idx * seg_samples
        end = start + seg_samples
        seg = audio[start:end]
        if len(seg) < seg_samples:
            seg = pad_silence(seg, seg_samples)
            logger.warning(f"seg {idx+1} too short, padded")
        seg_name = f"{base_name}_{idx + 1:04d}"
        sf.write(os.path.join(output_dir, f"{seg_name}.wav"), seg, sample_rate)
        new_tn.append((seg_name, tn_items[idx][1]))
        new_itn.append((seg_name, itn_items[idx][1]))

    write_text_list(os.path.join(output_dir, f"{base_name}_text_tn.txt"), new_tn)
    write_text_list(os.path.join(output_dir, f"{base_name}_text_itn.txt"), new_itn)
    logger.info(f"split done: {len(new_tn)} segs -> {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Audio concat and split tool")
    sub = parser.add_subparsers(dest="stage")

    p1 = sub.add_parser("concat", help="Stage 1: group by duration and concat")
    p1.add_argument("--wav_scp", required=True)
    p1.add_argument("--text_tn", required=True, help="text_tn 文件路径")
    p1.add_argument("--text_itn", required=True, help="text_itn 文件路径")
    p1.add_argument("--wav2dur", required=True)
    p1.add_argument("--output_dir", required=True)
    p1.add_argument("--sr", type=int, default=24000)

    p2 = sub.add_parser("align", help="Stage 2: align recorded audio")
    p2.add_argument("--concat_wav", required=True)
    p2.add_argument("--recorded_wav", required=True)
    p2.add_argument("--output_wav", required=True)
    p2.add_argument("--search_range", type=float, default=30.0)
    p2.add_argument("--sr", type=int, default=SAMPLE_RATE)

    p3 = sub.add_parser("split", help="Stage 3: split aligned audio")
    p3.add_argument("--aligned_wav", required=True)
    p3.add_argument("--concat_tn_txt", required=True, help="Phase1 输出的 *_tn.txt")
    p3.add_argument("--concat_itn_txt", required=True, help="Phase1 输出的 *_itn.txt")
    p3.add_argument("--output_dir", required=True)
    p3.add_argument("--segment_sec", type=float, required=True)
    p3.add_argument("--sr", type=int, default=SAMPLE_RATE)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "concat":
        stage1_concat(args.wav_scp, args.text_tn, args.text_itn,
                      args.wav2dur, args.output_dir, args.sr)
    elif args.stage == "align":
        stage2_align(args.concat_wav, args.recorded_wav, args.output_wav,
                     args.search_range, args.sr)
    elif args.stage == "split":
        stage3_split(args.aligned_wav, args.concat_tn_txt, args.concat_itn_txt,
                     args.output_dir, args.segment_sec, args.sr)
    else:
        logger.error("please specify stage: concat / align / split")


if __name__ == "__main__":
#     # Phase 1
# python run_audio_cat_cut.py concat \
#     --wav_scp data/wav.scp \
#     --text_tn data/text_tn \
#     --text_itn data/text_itn \
#     --wav2dur data/wav2dur \
#     --output_dir output/concat

# # Phase 2（不变）
# python run_audio_cat_cut.py align \
#     --concat_wav output/concat/10s_01.wav \
#     --recorded_wav recorded/10s_01.wav \
#     --output_wav output/aligned/10s_01.wav

# # Phase 3
# python run_audio_cat_cut.py split \
#     --aligned_wav output/aligned/10s_01.wav \
#     --concat_tn_txt output/concat/10s_01_tn.txt \
#     --concat_itn_txt output/concat/10s_01_itn.txt \
#     --output_dir output/segments \
#     --segment_sec 10
    main()
