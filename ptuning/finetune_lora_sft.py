from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer
import transformers
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import json
import datasets
import os
from transformers import (
    AutoModel,
)

#一些参数的定义，也可以放到.sh文件里。
model_type = '/workspace/user_code/qjzcy/llm/glm6b/chatglm-6b'
train_data = "data/train_news.json"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_type, trust_remote_code=True)
PRE_SEQ_LEN = 128
LR=1e-4
max_source_length = 128
max_target_length =128
max_seq_length = 128
skip_overlength = False

#定义了一个名为 CastOutputToFloat 的类，继承自 nn.Sequential 类。该类重写了 forward 方法，用于将模型输出转换为浮点数
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

# 定义 preprocess 函数，这里将输入的特征转化为id的特征，将它们转化为我们想要的input_ids和长度格式
def preprocess(tokenizer, config, example, max_seq_length):
    # 获取输入和目标文本
    prompt = example["content"]
    target = example["summary"]

    # 将输入和目标文本编码为 ID 序列
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)

    # 将输入和目标文本的 ID 序列拼接起来，并添加 EOS 标记
    input_ids = prompt_ids + target_ids + [config.eos_token_id]

    # 返回 input_ids 和 prompt_ids 的长度
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


# 定义 read_jsonl 函数，用于读取 JSONL 文件，
# 调用preprocess 函数，将输入的特征转化为id的特征，将它们转化为我们想要的input_ids和长度格式
def read_jsonl(json_data, max_seq_length, skip_overlength=False):
    # 加载模型配置
    config = transformers.AutoConfig.from_pretrained(
        model_type, trust_remote_code=True, device_map='auto')

    # 初始化 input_ids_list、attention_mask_list 和 seqlen_list 列表
    input_ids_list = []
    attention_mask_list = []
    seqlen_list = []

    # 遍历 JSONL 文件中的每一行数据
    for line in json_data:
        # 将 JSON 字符串转换为 Python 对象
        line = json.loads(line)

        # 预处理数据并将其转换为特征
        feature = preprocess(tokenizer, config, line, max_seq_length)

        # 如果 skip_overlength 为 True，且特征的 input_ids 长度超过了 max_seq_length，则跳过该特征
        if skip_overlength and len(feature["input_ids"]) > max_seq_length:
            continue

        # 将特征的 input_ids 截断到 max_seq_length 长度
        feature["input_ids"] = feature["input_ids"][:max_seq_length]

        # 将 input_ids 和 seq_len 添加到列表中
        input_ids_list.append(feature["input_ids"])
        seqlen_list.append(feature["seq_len"])

    # 返回 input_ids 和 seq_len 字典
    return {"input_ids": input_ids_list,  "seq_len": seqlen_list}

# 定义 data_collator 函数，使用read_jsonl处理好的结果，按需求将特征转换为模型lable，
# 注意这里输入的特征是已经转化为id的特征，将它们转化为我们想要的输入和lable格式
def data_collator(features: list) -> dict:
    # 计算每个特征的 input_ids 长度
    len_ids = [len(feature["input_ids"]) for feature in features]

    # 找到最长的 input_ids 长度
    longest = max(len_ids)

    # 初始化 input_ids 和 labels_list 列表
    input_ids = []
    labels_list = []

    # 遍历特征，根据需要制作我们的input和lable，
    # lable长度按seq_len截断，其余部分用[-100] 补齐，注意需要保证lable和输入长短一致
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)

    # 将 input_ids 和 labels_list 转换为张量
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    # 返回 input_ids 和 labels 字典
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# 定义 ModifiedTrainer 类，继承自 Trainer 类，保存有梯度变化的模型参数
class ModifiedTrainer(Trainer):
    # 重写 compute_loss 方法，计算模型的损失
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    # 重写 save_model 方法，保存模型
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # 保存有梯度变化的模型参数
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    writer = SummaryWriter()

    # 打开 train.json 文件
    json_data = open(train_data)

    # 读取数据集并转换为 Dataset 对象
    dataset = read_jsonl(json_data, max_seq_length, skip_overlength)
    train_dataset = datasets.Dataset.from_dict(dataset)

    # 加载预训练模型
    model = AutoModel.from_pretrained(model_type, load_in_8bit=True, trust_remote_code=True, device_map='auto')

    # 配置模型支持梯度检查点
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()

    # 配置模型支持输入梯度
    model.enable_input_require_grads()

    # 将 lm_head 层的输出转换为浮点数
    model.lm_head = CastOutputToFloat(model.lm_head)

    # 禁用模型缓存
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # 配置 Lora 模型的参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=8,
        lora_alpha=32, lora_dropout=0.1,
    )

    # 获取 Lora 模型
    model = get_peft_model(model, peft_config)

    # 配置模型支持并行计算
    model.is_parallelizable = True
    model.model_parallel = True

    # 配置训练参数
    training_args = TrainingArguments(
        "output",
        fp16=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        max_steps=1500,
        logging_steps=50,
        remove_unused_columns=False,
        seed=0,
        data_seed=0,
        group_by_length=False,
    )

    # 创建 ModifiedTrainer 对象并开始训练
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()

    # 保存模型
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
