#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并三个数据集并打乱顺序
输入：harmful.json、test_benign_alpaca.json、youdao_alpaca.json
输出：merged_alpaca.json
功能：
1. 读取三个 JSON 数据集
2. 合并为一个列表
3. 随机打乱顺序（设置随机种子保证可复现）
4. 保持每条样本内容不变，输出为标准 JSON 格式
"""

import json
import random
import os

# 输入文件列表
INPUT_FILES = [
    "harmful.json",
    "test_benign_alpaca.json",
    "youdao_alpaca.json"
]

OUTPUT_FILE = "merged_alpaca.json"
RANDOM_SEED = 42  # 固定随机种子，保证每次打乱结果一致


def merge_and_shuffle():
    all_data = []

    # 读取并合并三个数据集
    for file_name in INPUT_FILES:
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"已加载 {file_name}: {len(data)} 条样本")

    print(f"合并后总样本数: {len(all_data)}")

    # 随机打乱顺序
    random.seed(RANDOM_SEED)
    random.shuffle(all_data)

    # 写入输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"打乱完成，已保存至: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    merge_and_shuffle()
