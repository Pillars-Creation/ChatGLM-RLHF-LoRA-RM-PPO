from transformers import AutoModel, AutoTokenizer
import os

model_path = "../chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

# 设置只有第一张GPU可见

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2
max_length = 1240
top_p = 0.7
temperature = 0.95
history = []
string = ''
with open("../data/input500.txt", "r", encoding="utf-8") as f:
    # 逐行读取文件内容
    for line in f:
        string = string + line
    query = string + "<eop>上面每行是一篇文章标题，从中挑选出和体育相关的标题，不要自己扩展"
    response = model.chat(tokenizer, query, history)
    print("输出：", response[0])

# string = '25岁男性程序员喜欢什么样的文章'
# response = model.chat(tokenizer, string, history)
# print('输入：'+string)
# print('输出：'+response[0])
# print('')

# with open("input500.txt", "r", encoding="utf-8") as f:

#    for line in f:#     逐行读取文件内容
#         line = line.strip()#         去除行末的换行符
# #         构造输入字符串
# #        input_string = line + "看了这篇新闻的人最有可能喜欢上面哪三篇"
# #         生成标题
#         response = model.chat(tokenizer, line, history)
# #         输出结果
#         print("输入：",line)
#         print("输出：", response[0])

