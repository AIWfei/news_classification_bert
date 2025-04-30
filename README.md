# 基于BERT的中文新闻文本分类项目

本项目使用PyTorch对BERT-base-chinese模型进行微调，用于复旦新闻数据集的多类别文本分类任务。
## 目录结构
├── dataset/\
│ └── fudan_news_test.csv # 测试数据集\
│ └── fudan_news_train.csv # 训练数据集\
├── src/\
│ ├── dataset_class.py # 训练用数据集类\
│ ├── train.py # 训练脚本\
│ ├── test.py # 测试脚本\
│ ├── news_train_logs/ # tensorboard缓存目录\
│ └── zh_cls_fudan-news_model/ # 模型保存目录\
├── README.md/ # 文档\
└── requirements.txt # 依赖列表\
## 训练步骤
1. 数据加载
**使用工具**：Pandas  
```python
import pandas as pd
dataset = pd.read_csv('../dataset/fudan_news_train.csv')
```
2. 数据预处理
**使用工具**：Pandas + 原生Python
```python
# 创建类别编码映射
all_categories = sorted(list({c for row in dataset['category'] for c in eval(row)}))
category_to_id = {c:i for i,c in enumerate(all_categories)}
# 标签编码
dataset['label'] = dataset['output'].map(category_to_id)
```
3. 数据集划分
**使用工具**：scikit-learn
```python
import pandas as pd
dataset = pd.read_csv('../dataset/fudan_news_train.csv')
```
4. 文本编码
**使用工具**：HuggingFace Tokenizer bert模型分词器
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)
```
5. 数据集封装
**使用工具**：PyTorch Dataset
```python
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
```
6. 数据加载器
**使用工具**：PyTorch DataLoader
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
```
7. 模型初始化
**使用工具**：HuggingFace Transformers
```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=len(all_categories),
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
)
```
8. 训练配置
**使用工具**：PyTorch
```python
# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 学习率调度器
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=len(train_loader)*3
)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
```
9. 训练循环
**使用工具**：PyTorch + tqdm
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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
```
10. 模型评估
**使用工具**：PyTorch
```python
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
    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total
```
11. 模型保存
**使用工具**：HuggingFace Transformers
```python
model.save_pretrained('zh_cls_fudan-news_model')
tokenizer.save_pretrained('zh_cls_fudan-news_model')
```
12. 训练可视化
**使用工具**：TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
writer.add_scalar('train_loss', loss.item(), global_step)
writer.close()
```
## 快速开始

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 基础训练（默认参数）
python src/train.py

# 启动TensorBoard（自动生成日志）
tensorboard --logdir=news_train_logs
