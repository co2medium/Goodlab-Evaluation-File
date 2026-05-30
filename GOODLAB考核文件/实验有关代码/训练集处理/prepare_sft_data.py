import os
import sys
import json
import random
from pathlib import Path

# ==================== 配置区域 ====================
BASE_DIR = "."  # 数据文件所在目录

CVALUES_FILE = "train.jsonl"
BELLE_FILE   = "Belle_open_source_0.5M.json"
OUTPUT_FILE  = "sft_mixed_alpaca.json"

# 混合后总样本数控制（约 30k）
TARGET_TOTAL = 30000

# 比例配置：Belle 混入量 = CValues 数量 × BELLE_MULTIPLIER
# 推荐 2.0（1:2），安全数据占比约 33%
BELLE_MULTIPLIER = 2.0
BELLE_MAX = 200000   # Belle 采样上限
BELLE_MIN = 10000    # Belle 采样下限

RANDOM_SEED = 42
# ==================================================

CVALUES_PATH = os.path.join(BASE_DIR, CVALUES_FILE)
BELLE_PATH   = os.path.join(BASE_DIR, BELLE_FILE)
OUTPUT_PATH  = os.path.join(BASE_DIR, OUTPUT_FILE)


def process_cvalues(safety_target):
    """
    CValues 处理：
    1. 严格过滤：只保留 pos_type == "拒绝&正向建议"
    2. 按 prompt 去重，同一 prompt 保留 output 最长的一条（通常更详尽）
    3. 随机采样至 safety_target 条（控制总数据量）
    4. 转换为 Alpaca 格式
    """
    if not os.path.exists(CVALUES_PATH):
        print(f"[Error] 未找到 CValues 文件: {CVALUES_PATH}")
        sys.exit(1)

    records = {}  # prompt -> pos_resp
    total_lines = 0
    kept_lines = 0

    with open(CVALUES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt   = item.get("prompt", "").strip()
            pos_type = item.get("pos_type", "")
            pos_resp = item.get("pos_resp", "").strip()

            # 严格过滤：仅保留 "拒绝&正向建议"
            if pos_type != "拒绝&正向建议":
                continue

            kept_lines += 1

            if prompt not in records:
                records[prompt] = pos_resp
            else:
                # 同一 prompt 保留更长的正向建议（信息通常更丰富）
                if len(pos_resp) > len(records[prompt]):
                    records[prompt] = pos_resp

    print(f"[CValues] 原始总行数: {total_lines}")
    print(f"[CValues] 符合 '拒绝&正向建议' 的行数: {kept_lines}")
    print(f"[CValues] 去重后保留 prompt 数: {len(records)}")

    # 转换为 Alpaca 格式
    cvalues_data = [
        {"instruction": prompt, "input": "", "output": pos_resp}
        for prompt, pos_resp in records.items()
    ]

    # 若去重后超过目标数量，随机采样至 safety_target；否则全部保留
    if len(cvalues_data) > safety_target:
        random.seed(RANDOM_SEED)
        cvalues_data = random.sample(cvalues_data, safety_target)
        print(f"[CValues] 随机采样至目标数量: {safety_target} 条")
    else:
        print(f"[CValues] 去重后数量不足目标，保留全部: {len(cvalues_data)} 条")

    return cvalues_data


def load_belle():
    """加载 Belle 数据，兼容标准 JSON 数组与 JSONL 两种格式"""
    if not os.path.exists(BELLE_PATH):
        print(f"[Error] 未找到 Belle 文件: {BELLE_PATH}")
        sys.exit(1)

    with open(BELLE_PATH, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
        if not raw:
            print("[Error] Belle 文件为空")
            sys.exit(1)

        if raw.startswith('['):
            data = json.loads(raw)
        else:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return data


def process_belle(cvalues_count):
    """
    根据 CValues 数量计算 Belle 采样数并随机选取
    公式：min(max(cvalues_count * MULTIPLIER, MIN), MAX, len(belle_all))
    """
    belle_all = load_belle()
    belle_total = len(belle_all)

    target = int(cvalues_count * BELLE_MULTIPLIER)
    sample_count = max(BELLE_MIN, min(target, BELLE_MAX, belle_total))

    random.seed(RANDOM_SEED)
    sampled = random.sample(belle_all, sample_count)

    # 统一转换为标准 Alpaca 格式，确保字段为字符串并去除首尾空白
    formatted = []
    for item in sampled:
        formatted.append({
            "instruction": str(item.get("instruction", "")).strip(),
            "input":       str(item.get("input", "")).strip(),
            "output":      str(item.get("output", "")).strip()
        })

    print(f"[Belle] 原始总样本数: {belle_total}")
    print(f"[Belle] 本次采样混入数: {sample_count}")
    return formatted


def main():
    print("=" * 60)
    print("开始构建安全 SFT 混合数据集 (Alpaca 格式)")
    print(f"目标总样本数: ~{TARGET_TOTAL}")
    print("=" * 60)

    # 根据目标总数和比例计算安全数据目标数量
    # 比例 1 : BELLE_MULTIPLIER，安全数据占比 = 1 / (1 + MULTIPLIER)
    safety_target = int(TARGET_TOTAL / (1 + BELLE_MULTIPLIER))

    # 1. 处理 CValues（过滤、去重、采样至目标数量）
    cvalues_data = process_cvalues(safety_target)
    if not cvalues_data:
        print("[Warning] CValues 过滤后为空，请检查 pos_type 字段是否匹配")
        sys.exit(1)

    # 2. 处理 Belle（按 CValues 实际数量 × 2.0 采样）
    belle_data = process_belle(len(cvalues_data))

    # 3. 合并并打乱顺序
    combined = cvalues_data + belle_data
    random.seed(RANDOM_SEED)
    random.shuffle(combined)

    # 4. 保存为 JSON 数组（标准 Alpaca 格式）
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # 5. 统计信息
    total = len(combined)
    safety_num = len(cvalues_data)
    general_num = len(belle_data)
    safety_ratio = safety_num / total * 100 if total > 0 else 0

    print("=" * 60)
    print("数据集构建完成")
    print("=" * 60)
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"总样本数: {total}")
    print(f"  - 安全数据 (CValues): {safety_num} 条 ({safety_ratio:.1f}%)")
    print(f"  - 通用数据 (Belle):   {general_num} 条 ({100 - safety_ratio:.1f}%)")
    print("=" * 60)
    print("LlamaFactory dataset_info.json 配置示例：")
    print(json.dumps({
        "safety_sft_mix": {
            "file_name": OUTPUT_FILE,
            "formatting": "alpaca"
        }
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
