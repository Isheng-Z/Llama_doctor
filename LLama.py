import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
import warnings
from flask import Flask, request, jsonify
import sys
from flask_cors import CORS
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

# 使用 LlamaTokenizer 加载模型
tokenizer = LlamaTokenizer.from_pretrained('./Llama-2-7b-chat-hf')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用EOS作为填充标记
model.config.pad_token_id = tokenizer.pad_token_id


# 定义 Llama-2 专用对话模板（支持多轮对话）
def build_prompt(history):
    system_prompt = "<<SYS>>\nYou are a helpful medical assistant. Provide accurate and safe suggestions.\n<</SYS>>\n\n"
    prompt = "<s>[INST] " + system_prompt
    for user, assistant in history:
        prompt += f"{user} [/INST] {assistant} </s><s>[INST] "
    return prompt


# 创建 Flask 服务
app = Flask(__name__)
CORS(app)  # 允许所有来源的跨域请求
# 全局变量保存对话历史（仅适用于单用户场景）
conversation_history = []


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data.get('input', '')

    # 将用户输入加入对话历史（assistant回复为空）
    conversation_history.append((user_input, ""))

    try:
        # 构建完整提示
        full_prompt = build_prompt(conversation_history)
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # 防止超过上下文限制
        ).to("cuda")

        generate_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,  # 添加注意力掩码
            "max_new_tokens": 400,  # 适当增加回复长度
            "temperature": 0.7,  # 平衡创造性和准确性
            "top_p": 0.9,
            "repetition_penalty": 1.1,  # 防止重复
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id  # 确保使用正确的 pad token
        }

        # 调用大模型生成回复
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

            # 打印完整的生成输出（方便调试）
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("完整生成输出：", full_output)
            sys.stdout.flush()

            # 截取新增部分作为回复
            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            print("生成回复：", generated_text)
            sys.stdout.flush()

        # 更新对话历史中最后一轮（填入生成的回复）
        conversation_history[-1] = (user_input, generated_text)
        return jsonify({'output': generated_text})
    except Exception as e:
        # 出现异常时从历史中移除这一轮
        conversation_history.pop()
        print("发生错误：", str(e))
        sys.stdout.flush()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 监听所有网卡，方便局域网内调用
    app.run(host='0.0.0.0', port=5001)
