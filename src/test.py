import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 加载验证集
val_dataset_path = '../dataset/fudan_news_test.csv'  # 验证集路径
val_df = pd.read_csv(val_dataset_path)
dataset = pd.read_csv('../dataset/fudan_news_train.csv')
# 统计所有类别
all_categories = set()
for categories in dataset['category']:
    # 假设category列是字符串形式的列表，例如 "[cat1, cat2]"
    categories = eval(categories)  # 将字符串转换为列表
    all_categories.update(categories)

# 将类别转换为列表并排序
all_categories = sorted(list(all_categories))

# 创建类别到编码的映射
category_to_id = {category: idx for idx, category in enumerate(all_categories)}

# 加载训练好的模型和分词器
model_path = 'zh_cls_fudan-news_model'  # 替换为你的模型路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义函数：对单一文本进行预测
def predict_single_text(text, threshold=0.5):
    # 将文本编码为模型输入
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    inputs = {key: val.to(device) for key, val in inputs.items()}  # 将输入数据移动到设备

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 使用 sigmoid 将 logits 转换为概率
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # 根据阈值生成预测结果
    predicted_labels = [all_categories[i] for i, prob in enumerate(probs) if prob > threshold]

    # 找到概率最大的标签
    max_prob_index = probs.argmax()
    max_prob_label = all_categories[max_prob_index]
    max_prob_value = probs[max_prob_index]

    return predicted_labels, probs, max_prob_label, max_prob_value

# 示例：对单一文本进行预测
text = "这是一条测试新闻，内容涉及体育和科技。"  # 替换为你要验证的文本
predicted_labels, probs, max_prob_label, max_prob_value = predict_single_text(val_df.iloc[3]['text'])
print(val_df.iloc[3]['category'])
# 输出结果
print("输入文本:", text)
print("预测类别:", predicted_labels)
print("各类别概率:", {category: prob for category, prob in zip(all_categories, probs)})
print("概率最大的标签:", max_prob_label)
print("最大概率值:", max_prob_value)