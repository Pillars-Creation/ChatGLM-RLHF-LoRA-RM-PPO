from transformers import AutoModel, AutoTokenizer
import gradio as gr
import time
import math

tokenizer = AutoTokenizer.from_pretrained("../chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("../chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2
max_length = 1024
top_p = 0.7
temperature = 0.95
history = []
string = ''
with open("article.txt", "r", encoding="utf-8") as f:
    # 逐行读取文件内容
    for line in f:
        string = string+ line
string = string +"上面每行是一篇标题，选三篇推荐给45岁汽车司机"

#with open("input.txt", "r", encoding="utf-8") as f:
    # 逐行读取文件内容
#    for line in f:
        # 去除行末的换行符
#        line = line.strip()
        # 构造输入字符串
#        input_string = line + "看了这篇新闻的人最有可能喜欢上面哪三篇"
        # 生成标题
#        response = model.chat(tokenizer, input_string, history)
        # 输出结果
#        print("输入：",line," 输出：", response)
#        print('')
response = model.chat(tokenizer, string, history)
print(response)
print('')