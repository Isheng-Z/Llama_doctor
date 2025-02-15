import torch
from peft import PeftConfig, PeftModel
from transformers import  AutoModelForCausalLM, BitsAndBytesConfig
import warnings
from transformers import LlamaTokenizer
from CheckInput import classify_text
from Chat_Module import build_prompt
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
base_model = AutoModelForCausalLM.from_pretrained(
    "./Llama-2-7b-chat-hf",  # 本地路径
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True  # 强制使用本地文件
)
# 2. 加载LoRA适配器
peft_model_id = "llama2-7b-medical-lora"  # 替换为实际路径
config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(
    base_model,
    peft_model_id,
    is_trainable=False  # 推理模式（如需继续训练设为True）
)

# 3. 合并权重（可选，提升推理速度）
model = model.merge_and_unload()
# 加载并配置分词器
# tokenizer = AutoTokenizer.from_pretrained(
#     "./Llama-2-7b-chat-hf",
#     use_fast=True,
#     padding_side="right",  # 确保填充方向正确
#     local_files_only=True
# )




# 使用 LlamaTokenizer 加载模型
tokenizer = LlamaTokenizer.from_pretrained('./Llama-2-7b-chat-hf')




# 关键修复：设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用EOS作为填充标记

# 同步模型配置
model.config.pad_token_id = tokenizer.pad_token_id


# 初始化对话历史
dialogue_history = []
initial_instruction = "Please ask me about your health concerns."

#对话轮次
count = 0

# 启动对话
print("医疗助手已启动（输入 'exit' 退出）\n")
print("AI：您好，我是您的在线问诊助手，请问您主要是关于哪方面的问题呢？比如呼吸系统、消化系统、心血管、神经系统、皮肤问题等")

while True:
    # 获取用户输入
    user_input = input("\n用户：")
    if user_input.lower() in ['exit', 'quit']:
        print("对话结束")
        break

    if classify_text(user_input):
        print("AI: 我是医疗助手,请说明您的健康方面疑问！")
        continue


    # 添加用户输入到历史
    dialogue_history.append((user_input, ""))

    try:
        # 构建完整提示
        full_prompt = build_prompt(dialogue_history,count)
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