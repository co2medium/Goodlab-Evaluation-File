import json

# 自动读取当前目录的 data.jsonl
INPUT_FILE = "test.jsonl"
# 自动输出到当前目录的 alpaca 格式 json
OUTPUT_FILE = "test_alpaca_data.json"

alpaca_list = []

# 逐行读取 JSONL 并转换
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        # 标准 Alpaca 格式 + 完整保留安全标签
        alpaca_item = {
            "instruction": item.get("prompt", ""),
            "input": "",
            "output": item.get("response", ""),
            # 保留所有安全字段
            "is_safe": item.get("is_safe"),
            "category": item.get("category", {}),
        }

        alpaca_list.append(alpaca_item)

# 写入 JSON 文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(alpaca_list, f, ensure_ascii=False, indent=2)

print(
    f"✅ 转换完成！\n输入文件：{INPUT_FILE}\n输出文件：{OUTPUT_FILE}\n总计：{len(alpaca_list)} 条数据"
)
