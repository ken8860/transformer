import json
import time
import random  # 导入随机库
from openai import OpenAI

# --- 核心配置 ---
client = OpenAI(
    api_key="sk-230c3f5112c1436aa738d0056adfa690",
    base_url="https://api.deepseek.com"
)

INPUT_FILE = "vocabulary.txt"
EN_OUTPUT = "english_20000.txt"
ZH_OUTPUT = "chinese_20000.txt"
WORDS_PER_BATCH = 40  # 稍微减小每批单词数，提高单句质量
SENTENCES_PER_BATCH = 100  # 每批要求生成的句子数量
TARGET_TOTAL_SENTENCES = 20000


def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def generate_sentences(words_batch, current_total):
    words_str = ", ".join(words_batch)

    # 随机变换语境要求，增加多样性
    contexts = ["商务职场", "前沿科技", "日常文学", "法律医疗", "哲学艺术", "学术论文"]
    selected_context = random.choice(contexts)

    prompt = f"""
    你是一个专业的翻译语料库专家。当前任务进度：已完成 {current_total}/{TARGET_TOTAL_SENTENCES}。
    我有以下单词：[{words_str}]

    任务要求：
    1. 编写 80 到 100 句高质量中英文对照句子。
    2. 侧重于【{selected_context}】语境，但也要兼顾其他场景。
    3. 严禁直接重复简单的定义，要编写具有复杂语法结构（从句、被动语态等）的长难句。
    4. 确保每个单词至少被使用 2 次。
    5. 严格返回 JSON 数组格式：[{"{"}"en": "...", "zh": "..."{"}"}, ...]
    不要文字说明。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8  # 提高随机性
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"❌ 出错: {e}")
        return None


def main():
    all_words = load_words(INPUT_FILE)
    current_count = 0

    print(f"🚀 开始冲刺 20,000 句任务...")

    with open(EN_OUTPUT, 'a', encoding='utf-8') as ef, \
            open(ZH_OUTPUT, 'a', encoding='utf-8') as zf:

        # 只要总数没达标，就不断打乱词表进行新一轮抽取
        while current_count < TARGET_TOTAL_SENTENCES:
            random.shuffle(all_words)  # 核心：每轮打乱，保证单词组合全新

            for i in range(0, len(all_words), WORDS_PER_BATCH):
                if current_count >= TARGET_TOTAL_SENTENCES:
                    break

                batch = all_words[i: i + WORDS_PER_BATCH]
                print(f"进度: {current_count}/{TARGET_TOTAL_SENTENCES} | 正在处理: {batch[0]}...")

                result = generate_sentences(batch, current_count)

                if result:
                    sentences = []
                    if isinstance(result, list):
                        sentences = result
                    elif isinstance(result, dict):
                        sentences = list(result.values())[0]

                    batch_written = 0
                    for s in sentences:
                        if 'en' in s and 'zh' in s:
                            ef.write(s['en'].strip().replace('\n', ' ') + "\n")
                            zf.write(s['zh'].strip().replace('\n', ' ') + "\n")
                            batch_written += 1
                            current_count += 1

                    print(f"✅ 本批次成功写入 {batch_written} 条，总计 {current_count} 条")
                    ef.flush()  # 实时刷入硬盘，防止断电
                    zf.flush()

                time.sleep(1.5)  # 给 API 喘息时间

    print(f"🎉 任务圆满完成！共生成 {current_count} 句。")


if __name__ == "__main__":
    main()