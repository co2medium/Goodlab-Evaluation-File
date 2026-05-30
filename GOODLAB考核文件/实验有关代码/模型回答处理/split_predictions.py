#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拆分 merged_predictions.json 脚本
按 category 分组，按 id 排序，并去除无用字段
"""

import json
import os
from collections import defaultdict

# 配置
INPUT_FILE = "merged_predictions.json"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 需要移除的字段
FIELDS_TO_REMOVE = {"input", "output", "label", "prompt_raw"}


def main():
    # 读取源文件
    input_path = os.path.join(OUTPUT_DIR, INPUT_FILE)
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误：{e}")
            return

    # 确保是列表
    if not isinstance(data, list):
        print("错误：文件内容应为 JSON 列表")
        return

    # 按 category 分组
    groups = defaultdict(list)
    for item in data:
        category = item.get("category", "unknown")
        groups[category].append(item)

    # 处理每个分组
    for category, items in groups.items():
        # 按 id 排序
        items.sort(key=lambda x: x.get("id", 0))

        # 移除无用字段
        cleaned_items = []
        for item in items:
            cleaned = {
                k: v for k, v in item.items()
                if k not in FIELDS_TO_REMOVE
            }
            cleaned_items.append(cleaned)

        # 写入文件
        output_filename = f"{category}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_items, f, ensure_ascii=False, indent=2)

        print(f"已生成 {output_filename}，共 {len(cleaned_items)} 条记录")

    print("拆分完成！")


if __name__ == "__main__":
    main()
