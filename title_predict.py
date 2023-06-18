from transformers import AutoModel, AutoTokenizer
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
response = model.chat(tokenizer, string, history)
print(response)
print('')