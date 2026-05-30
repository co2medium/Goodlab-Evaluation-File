#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 LlamaFactory 预测结果与原始元数据
功能：将 generated_predictions.jsonl 与原始 merged_alpaca.json 按顺序合并，
      恢复 category、id、instruction 等字段
"""

import json
import re
from pathlib import Path

# ==================== 配置（在同一目录下，无需修改路径）====================
ORIGINAL_JSON = "merged_alpaca.json"  # 原始测试集（含 category/id）
PREDICTION_JSONL = "generated_predictions.jsonl"  # LlamaFactory 输出
OUTPUT_JSONL = "merged_predictions.jsonl"  # 合并后输出（用于后续裁判）
OUTPUT_JSON = "merged_predictions.json"  # 合并后输出（数组格式，方便查看）


# =======================================================================

def clean_text(text: str) -> str:
    """统一空白字符，用于模糊匹配"""
    if not text:
        return ""
    # 将各种换行、制表符统一为空格，再去掉多余空白
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\t', ' '))
    return text.strip()


def main():
    # 1. 读取原始 JSON
    with open(ORIGINAL_JSON, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    print(f"[原始数据] 共 {len(original_data)} 条")

    # 2. 读取预测结果 JSONL
    predictions = []
    with open(PREDICTION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            predictions.append(json.loads(line))
    print(f"[预测结果] 共 {len(predictions)} 条")

    # 数量检查
    if len(original_data) != len(predictions):
        print(f"[!] 警告: 数量不一致 ({len(original_data)} vs {len(predictions)})")
        print(f"    将按最小数量处理，请检查是否有样本生成失败")

    # 3. 按顺序合并，并进行内容校验
    merged = []
    mismatch_count = 0

    for i, (orig, pred) in enumerate(zip(original_data, predictions)):
        orig_instr = orig.get("instruction", "").strip()
        orig_input = orig.get("input", "").strip()

        # 构建完整查询文本（与 LlamaFactory 拼接逻辑一致）
        full_query = orig_instr
        if orig_input:
            full_query += "\n" + orig_input

        # 清洗用于校验
        query_norm = clean_text(full_query)
        prompt_norm = clean_text(pred.get("prompt", ""))

        # 校验：instruction 核心内容是否出现在 prompt 中
        # 允许短文本（<<10字符）跳过严格校验，避免误判
        is_match = (len(query_norm) < 10) or (query_norm in prompt_norm) or (prompt_norm in query_norm)

        if not is_match:
            # 二次校验：去除所有非字母数字字符后再比
            q_simple = re.sub(r'[^\w\u4e00-\u9fff]', '', query_norm)  # 保留中文
            p_simple = re.sub(r'[^\w\u4e00-\u9fff]', '', prompt_norm)
            is_match = (len(q_simple) < 5) or (q_simple in p_simple) or (p_simple in q_simple)

            if not is_match:
                mismatch_count += 1
                print(f"  [!] 第 {i + 1} 条内容校验不通过，请人工检查:")
                print(f"      原始: {orig_instr[:50]}...")
                print(f"      Prompt: {pred.get('prompt', '')[:50]}...")

        # 合并记录：保留所有原始字段 + 预测字段
        merged_record = {
            "id": orig.get("id"),
            "category": orig.get("category"),
            "instruction": orig_instr,
            "input": orig.get("input", ""),
            "output": orig.get("output", ""),
            "prompt_raw": pred.get("prompt"),  # LlamaFactory 格式化后的完整 prompt
            "predict": pred.get("predict"),  # 模型生成的回答（核心！）
            "label": pred.get("label", "")  # LlamaFactory 自带的 label（通常为空）
        }
        merged.append(merged_record)

    # 4. 保存 JSONL（每行一条，方便后续流式处理）
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 5. 保存 JSON Array（方便人工查看和调试）
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[合并完成] 共处理 {len(merged)} 条")
    if mismatch_count > 0:
        print(f"[!] 其中 {mismatch_count} 条内容校验存在偏差，建议人工抽检")
    else:
        print(f"[✓] 全部 {len(merged)} 条顺序与内容校验通过")
    print(f"[输出文件] {OUTPUT_JSONL} 和 {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
