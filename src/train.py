import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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


# 应用编码函数
dataset['label'] = dataset['output'].map(category_to_id)
# 划分训练集和测试集，确保测试集样本均衡
train_df, test_df = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42,
    stratify=dataset['label']  # 按 label 分层抽样
)

# 检查训练集和测试集的样本分布
print("训练集样本分布:")
print(train_df['label'].value_counts())
print("\n测试集样本分布:")
print(test_df['label'].value_counts())


# 加载BERT的分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 对训练集和测试集的文本进行编码
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)



class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 创建训练集和测试集的 Dataset
train_dataset = NewsDataset(train_encodings, train_df['label'].tolist())
test_dataset = NewsDataset(test_encodings, test_df['label'].tolist())


# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)



# 加载 BERT 模型
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(all_categories))
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=len(all_categories),
    hidden_dropout_prob=0.2,  # 添加 Dropout
    attention_probs_dropout_prob=0.2  # 添加 Dropout
)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
# optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 定义学习率调度器
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#设置训练网络的一些参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
# 训练循环
epochs = 3
for epoch in range(epochs):
    print("------第{}轮训练开始------".format(epoch + 1))
    model.train()
    total_loss = 0

     # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    # 添加tensorboard
    writer = SummaryWriter("news_train_logs")
    for batch in progress_bar:
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算损失
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})

    # 打印平均损失
    avg_loss = total_loss / len(train_loader)
    # print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    print(f"第 {epoch + 1} 轮, 该轮平均损失: {avg_loss}")


    # 验证集评估
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            print(logits)
            val_loss += loss_fn(logits, labels).item()

            # 计算准确率
            _, predicted = torch.max(logits, 1)
            print(predicted)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_test_step += 1
            print("Predicted:", predicted)
            print("Labels:", labels)
            print("Correct:", (predicted == labels).sum().item())
            print("Total:", labels.size(0))
    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total
    print(total)
    print("整体测试集上的Loss：{}".format(val_loss))
    print("整体测试集上的平均Loss：{}".format(avg_val_loss))
    print("整体测试集上的正确率：{}".format(accuracy))
    # print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
    writer.add_scalar("test_loss", val_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy, total_test_step)
    #保存每轮模型
    # 模型保存 生成以下文件：pytorch_model.bin，config.json
    model.save_pretrained('zh_cls_fudan-news_model')
    # 分词器保存
    tokenizer.save_pretrained('zh_cls_fudan-news_model')
    print("模型已保存")

writer.close()