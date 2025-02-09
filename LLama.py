import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings


# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*flash attention.*")

# 量化配置（显存优化核心）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # 二次量化节省显存
    bnb_4bit_quant_type="nf4",  # 最优量化算法
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型（首次运行需等待下载）
model = AutoModelForCausalLM.from_pretrained(
    "./Llama-2-7b-chat-hf",  # 本地路径
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True  # 强制使用本地文件
)

# 加载并配置分词器
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    use_fast=True,
    padding_side="right"  # 确保填充方向正确
)

# 关键修复：设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用EOS作为填充标记

# 同步模型配置
model.config.pad_token_id = tokenizer.pad_token_id


# 定义Llama-2专用对话模板（支持多轮对话）
def build_prompt(history):
    system_prompt = "<<SYS>>\nYou are a helpful medical assistant. Provide accurate and safe suggestions.\n<</SYS>>\n\n"
    prompt = "<s>[INST] " + system_prompt
    for user, assistant in history:
        prompt += f"{user} [/INST] {assistant} </s><s>[INST] "
    return prompt


# 初始化对话历史
dialogue_history = []
initial_instruction = "Please ask me about your health concerns."

# 启动对话
print("医疗助手已启动（输入 'exit' 退出）\n")
print("AI：您好，我是医疗助手。请问您有什么健康方面的疑问？")

while True:
    # 获取用户输入
    user_input = input("\n用户：")

    if user_input.lower() in ['exit', 'quit']:
        print("对话结束")
        break

    # 添加用户输入到历史
    dialogue_history.append((user_input, ""))

    try:
        # 构建完整提示
        full_prompt = build_prompt(dialogue_history)
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # 防止超过上下文限制
        ).to("cuda")

        # 生成参数配置
        generate_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,  # 关键修复：添加注意力掩码
            "max_new_tokens": 400,  # 适当增加医疗建议长度
            "temperature": 0.7,  # 平衡创造性和准确性
            "top_p": 0.9,
            "repetition_penalty": 1.1,  # 防止重复
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id  # 确保使用正确pad token
        }

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )

        # 更新对话历史
        dialogue_history[-1] = (user_input, response)

        # 打印回复（带格式优化）
        print("\nAI：" + response.replace("\n", "\n    "))

    except Exception as e:
        print(f"\n发生错误：{str(e)}")
        dialogue_history.pop()  # 移除无效对话轮次