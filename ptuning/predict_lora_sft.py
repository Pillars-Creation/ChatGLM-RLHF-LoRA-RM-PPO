import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import random
import pandas as pd
import difflib

results_rm = []
# 读取数据文件
df = pd.read_csv('./data/title_cat_1000.csv', sep='\t', header=None, names=['title', 'category'])

# 加载预训练模型和分词器
model_name = "../../chatglm-6b/"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, device_map='auto')

# 读取lora sft模型参数
model_path = "./path_to_sft_checkpoint"
peft_model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.long)
peft_model.eval()
results_rm = []

# 构造instruction和input
def construct_input(article_list, cat):
    random.shuffle(article_list)
    instruction = '{}上面文章中找到{}相关的文章summary：'.format(article_list, cat)
    return instruction


# 判断answer结果是否与article_list中的元素相似
def is_similar(ans, article_list, similarity_threshold):
    for article in article_list:
        similarity = difflib.SequenceMatcher(None, ans, article).ratio()
        if similarity >= similarity_threshold:
            return True
    return False

# 批处理输入数据
input_batch = []
range_num =100
positive_num = 0
for i in range(100):
       random_articles = df.sample(n=31)
       random_article = random_articles.iloc[0]
       cat = random_article['category']
       article_list = [title + ' (' + cat + ')' for title, cat in zip(random_articles['title'], random_articles['category'])]
       input_str = construct_input(article_list, cat)
       input_ids = tokenizer.encode(input_str, return_tensors='pt').to('cuda')
       out = model.generate(
           input_ids=input_ids,
           max_length=1500,
           temperature=0
       )
       out = tokenizer.batch_decode(out, skip_special_tokens=True)
       answer = out[0].split('summary:')[1].strip()

       print('新闻：', input_str)
       print('answer :', answer)
       for ans in answer.split('\n'):
           similarity_threshold = 0.9  # 相似度阈值
           # 判断是否在input中且分类是否一致
           if is_similar(ans, article_list, similarity_threshold):
               positive_num = positive_num +1
               break
       print(i, 'accuracy:', positive_num / (i+1))






