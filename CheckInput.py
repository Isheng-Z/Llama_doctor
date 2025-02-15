from transformers import pipeline

# 初始化零样本分类器，使用 Facebook 的 BART-large-MNLI 模型
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_text(text):
    """
    对输入文本进行分类：
      如果文本被判定为与医疗相关，则返回 1；
      否则返回 0。
    """
    # 定义候选标签
    candidate_labels = [
        "non-healthcare",  # 非医疗相关
        "general healthcare",  # 一般医疗健康问题
        "medical",  # 医疗相关
        "healthcare question",  # 医疗疑问
        "sleep disorder",  # 睡眠障碍相关
        "mental health",  # 心理健康相关
        "nutrition",  # 营养咨询相关
        "emergency care"  # 急诊护理相关
    ]
    # 进行零样本分类
    result = classifier(text, candidate_labels)
    # 返回得分最高的标签所对应的类别
    best_label = result['labels'][0]
    return True if best_label == "non-healthcare" else False

