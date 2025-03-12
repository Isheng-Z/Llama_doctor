import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
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

# 1. 加载基础模型（首次运行需等待下载）
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

# 4. 加载并配置分词器
tokenizer = LlamaTokenizer.from_pretrained('./Llama-2-7b-chat-hf')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用EOS作为填充标记
model.config.pad_token_id = tokenizer.pad_token_id

# 初始化对话历史和轮次计数（仅适用于单用户场景）
dialogue_history = []
count = 0

# 创建 Flask 服务
app = Flask(__name__)
CORS(app)  # 允许所有来源的跨域请求


@app.route('/generate', methods=['POST'])
def generate():
    global dialogue_history, count
    data = request.get_json()
    user_input = data.get('input', '')

    # 检查输入内容（如不符合要求，返回提示信息）
    if classify_text(user_input):
        return jsonify({'output': '我是医疗助手，请告诉我您的健康问题！'})

    # 将用户输入加入对话历史（assistant回复暂为空）
    dialogue_history.append((user_input, ""))

    try:
        # 构建完整提示（可能包含多轮对话历史）
        full_prompt = build_prompt(dialogue_history, count)
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
            "attention_mask": inputs.attention_mask,  # 添加注意力掩码
            "max_new_tokens": 400,  # 适当增加医疗建议长度
            "temperature": 0.7,  # 平衡创造性和准确性
            "top_p": 0.9,
            "repetition_penalty": 1.1,  # 防止重复
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id  # 确保使用正确pad token
        }

        # 调用大模型生成回复
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
            # 截取新增部分作为回复
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )

        # 更新对话历史中最后一轮
        dialogue_history[-1] = (user_input, response)
        return jsonify({'output': response})
    except Exception as e:
        dialogue_history.pop()  # 移除无效对话轮次
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 监听所有网卡，方便局域网内调用
    app.run(host='0.0.0.0', port=5001)
