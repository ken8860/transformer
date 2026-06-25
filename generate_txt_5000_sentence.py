import json
import time
from openai import OpenAI

# --- 核心配置 ---
client = OpenAI(
    api_key="sk-230c3f5112c1436aa738d0056adfa690",  # 填入你在第一步申请的Key
    base_url="https://api.deepseek.com"  # DeepSeek 的官方接口地址
)

INPUT_FILE = "vocabulary.txt"
EN_OUTPUT = "english_5000.txt"
ZH_OUTPUT = "chinese_5000.txt"
WORDS_PER_BATCH = 80  # DeepSeek 建议每批处理 80-100 词，效果最稳


def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def generate_sentences(words_batch):
    words_str = ", ".join(words_batch)

    # 针对训练语料库优化的提示词
    prompt = f"""
    你是一个专业的翻译语料库生成专家。我有以下{len(words_batch)}个英文单词：
    [{words_str}]

    任务要求： 
    1. 编写约 120-150 句中英文对照的句子。
    2. 必须覆盖列表中所有的单词，且句子必须通顺、符合人类逻辑。
    3. 涵盖商业、科学、日常生活等多种语境。
    4. 严格按照 JSON 数组格式返回，格式如下：
    [
      {{"en": "The constituent elements of the mixture are unknown.", "zh": "这种混合物的组成成分尚不明确。"}},
      ...
    ]
    不要输出任何多余的文字说明，只输出 JSON。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 使用 DeepSeek V3 模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # 强制返回 JSON
            temperature=0.7
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"当前批次生成出错: {e}")
        return None


def main():
    all_words = load_words(INPUT_FILE)
    total_words = len(all_words)
    print(f"🚀 已加载 {total_words} 个单词，准备调用 DeepSeek 生成...")

    # 使用追加模式 'a'，防止中断后可以继续
    with open(EN_OUTPUT, 'a', encoding='utf-8') as ef, \
            open(ZH_OUTPUT, 'a', encoding='utf-8') as zf:

        for i in range(0, total_words, WORDS_PER_BATCH):
            batch = all_words[i: i + WORDS_PER_BATCH]
            print(f"正在处理进度: {i}/{total_words} (当前单词: {batch[0]}...)")

            result = generate_sentences(batch)

            if result:
                # 处理 DeepSeek 返回的 JSON 结构
                sentences = []
                if isinstance(result, list):
                    sentences = result
                elif isinstance(result, dict):
                    sentences = list(result.values())[0]

                count = 0
                for s in sentences:
                    if 'en' in s and 'zh' in s:
                        ef.write(s['en'].strip() + "\n")
                        zf.write(s['zh'].strip() + "\n")
                        count += 1
                print(f"✅ 成功写入 {count} 条高质量句子")

            # 稍微停顿，避免触发频率限制
            time.sleep(1)

    print(f"🎉 任务完成！结果已保存在 {EN_OUTPUT} 和 {ZH_OUTPUT}")


if __name__ == "__main__":
    main()