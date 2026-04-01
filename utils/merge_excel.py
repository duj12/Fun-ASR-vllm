import sys
import os
import pandas as pd
import glob


dirs = sys.argv[1]

dirs = dirs.split(',')

all_files=[]

for dir in dirs:
	path=dir
	all_files.extend(glob.glob(f"{path}/speaker_info/*.xlsx"))


print(all_files)

# 读取所有Excel文件并合并
combined_df = pd.concat([pd.read_excel(f) for f in all_files])

# 导出合并后的DataFrame到一个Excel文件
combined_df.to_excel(f"{path}/LAM_speaker_info.xlsx", index=False)