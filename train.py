from peft import LoraConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, LlamaTokenizer, DataCollatorForSeq2Seq
from trl import SFTTrainer
import torch
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM

# 加载本地数据集 disease_data.json
# 如果 JSON 文件格式为数组形式，可直接加载；如存在嵌套，可调整 field 参数
dataset = load_dataset("json", data_files="disease_data.json", split="train")
print(dataset)
print(dataset.column_names)  # 预期输出：['disease', 'description']

# 使用 LlamaTokenizer 加载模型的 tokenizer
tokenizer = LlamaTokenizer.from_pretrained('./Llama-2-7b-chat-hf')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS 作为填充标记

# 数据预处理：将 'description' 文本编码为模型输入格式
def preprocess_function(examples):
    # 如果希望将疾病名称和描述组合起来，可以如下处理：
    texts = [f"{disease}: {desc}" for disease, desc in zip(examples['disease'], examples['description'])]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=512)
    # return tokenizer(examples['description'], truncation=True, padding='max_length', max_length=512)

def add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example

# 对数据集进行预处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# LoRA 配置
lora_config = LoraConfig(
    r=8,                      # 秩维度
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 关键层选择
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 训练参数配置
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # 显存不足时降为1
    gradient_accumulation_steps=8,        # 保持总 batch 等效
    gradient_checkpointing=True,          # 显存换速度
    fp16=True,
    optim="paged_adamw_8bit",
    max_steps=1000,
    logging_steps=50,
    output_dir="./med_llama_finetune"
)

# 量化配置（显存优化核心）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # 二次量化节省显存
    bnb_4bit_quant_type="nf4",           # 最优量化算法
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载 Llama 模型（确保本地路径正确）
model = AutoModelForCausalLM.from_pretrained(
    "./Llama-2-7b-chat-hf",  # 本地模型路径
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True   # 强制使用本地文件
)

# 同步模型配置
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False  # 禁用缓存，确保 gradient checkpointing 正常运行

# 训练器设置（这里使用的是 Seq2Seq 的 data collator，可根据需求切换到 DataCollatorForCompletionOnlyLM）
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets,
    peft_config=lora_config,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

# 如果需要使用标记数据的 collator，可以取消下面代码块的注释，并注释上面的 trainer 配置
"""
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets,
    peft_config=lora_config,
    args=training_args,
    data_collator=DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="### Response",
        mlm=False
    )
)
"""

# 启动训练（预计约需4-6小时，根据实际数据量和硬件情况）
trainer.train()

# 保存适配器
trainer.save_model("llama2-7b-medical-lora")
