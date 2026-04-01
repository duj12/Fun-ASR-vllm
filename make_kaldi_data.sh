#!/usr/bin/env bash

audio_dir=$1
data_dir=$2
shift 2
formats=("$@")  # 剩余参数为音频格式列表，如 wav flac

echo "音频路径：$audio_dir"
echo "输出数据路径：$data_dir"
echo "支持的音频格式：${formats[*]}"
mkdir -p "$data_dir"

echo "# 准备转写所需的kaldi格式数据，至少包含wav.scp和text，过滤掉[0.5, 40]s之外的音频段"

# ---------- 构建 find 命令 ----------
find_cmd=(find "$audio_dir" -type f \( )
first=1
for fmt in "${formats[@]}"; do
    if [ $first -eq 1 ]; then
        find_cmd+=(-name "*.${fmt}")
        first=0
    else
        find_cmd+=(-o -name "*.${fmt}")
    fi
done
find_cmd+=(\))

# ---------- 生成 wav.scp ----------
"${find_cmd[@]}" | awk -F"/" '
{
    name=$NF
    sub(/\.[^.]+$/, "", name)
    print name "\t" $0
}' | sort > "$data_dir/wav.scp"

# ---------- 生成 text ----------
if [ ! -f "$data_dir/text" ]; then
    find "$audio_dir" -name "*.txt" -print0 | xargs -0 awk 1 > "$data_dir/text"
fi

# ---------- 生成 utt2spk ----------
"${find_cmd[@]}" | awk -F"/" '
{
    name=$NF
    sub(/\.[^.]+$/, "", name)
    print name "\t" $(NF-1)
}' | sort > "$data_dir/utt2spk"

# ---------- 计算时长 ----------
bash utils/wav_to_duration.sh --nj 48 "$data_dir/wav.scp" "$data_dir/wav2dur"

mkdir -p $data_dir/backup
mv $data_dir/*  $data_dir/backup
# 将时长超过40s的音频都过滤掉。
cat ${data_dir}/backup/wav2dur | awk '{if($2<=40 && $2>=0.5) print $0}' > ${data_dir}/wav2dur
for f in wav.scp text ; do
  perl utils/filter_scp.pl ${data_dir}/wav2dur ${data_dir}/backup/$f > ${data_dir}/$f
done
bash utils/fix_data.sh ${data_dir}   # fix wav.scp and text
perl utils/filter_scp.pl ${data_dir}/wav.scp ${data_dir}/backup/utt2spk > ${data_dir}/utt2spk
perl utils/utt2spk_to_spk2utt.pl  $data_dir/utt2spk > $data_dir/spk2utt
cp $data_dir/text $data_dir/text_punc