import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 加载预训练模型和分词器
model_name = "../../chatglm-6b/"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, load_in_8bit=False, trust_remote_code=True, device_map='auto')

# 读取lora模型参数
model_path = "./output"
model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.long)
model.eval()

# 输入新闻和输出
news_text = "content：联播调查·被拆掉的承重墙，专家：承重墙动不得，非承重墙尽量不破坏,summary: "
input_ids = tokenizer.encode(news_text, return_tensors='pt').long()
out = model.generate(
        input_ids=input_ids,
        max_length=150,
        temperature=0
    )
answer = tokenizer.decode(out[0])
print('新闻：', answer)

