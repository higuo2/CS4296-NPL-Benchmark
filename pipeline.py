import pandas as pd
import numpy as np
import re
import time
import psutil
import os
import nltk
import gc
import json
import boto3
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 配置区 ---
BUCKET_NAME = "你的S3桶名"  # 如果在本地测试，可以设为 None
FILE_KEY = "IMDB Dataset.csv"
INSTANCE_TYPE = "cpu" # 运行前手动修改，用于记录结果
USE_S3 = False # 本地测试设为 False, 传到 AWS 设为 True

# --- 监控函数 ---
def get_metrics():
    process = psutil.Process(os.getpid())
    return {
        "cpu_usage": psutil.cpu_percent(interval=0.1),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024)
    }

# --- 从 S3 加载数据 ---
def load_data():
    if USE_S3:
        print(f"Fetching data from S3 bucket: {BUCKET_NAME}...")
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
        # 注意：对于大文件，S3 读取建议也采用流式，这里为了兼容演示先读取
        return pd.read_csv(io.BytesIO(obj['Body'].read()), chunksize=5000)
    else:
        print("Loading data from local disk...")
        return pd.read_csv(FILE_KEY, chunksize=5000)

# --- 清洗函数 ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def main():
    print(f"=== Starting Benchmark on {INSTANCE_TYPE} ===")
    start_all = time.time()
    
    all_cleaned_reviews = []
    all_sentiments = []
    
    # 1 & 2. 分块加载与清洗
    try:
        reader = load_data()
        for i, chunk in enumerate(reader):
            chunk['review'] = chunk['review'].apply(clean_text)
            all_cleaned_reviews.extend(chunk['review'].tolist())
            all_sentiments.extend(chunk['sentiment'].tolist())
            print(f"Chunk {i+1} processed. Memory: {get_metrics()['memory_usage_mb']:.2f} MB")
            gc.collect()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. 特征提取
    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000, dtype=np.float32)
    X = tfidf.fit_transform(all_cleaned_reviews)
    y = [1 if s == 'positive' else 0 for s in all_sentiments]
    
    del all_cleaned_reviews
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 训练
    print("Training...")
    model = LogisticRegression(max_iter=1000)
    t_start = time.time()
    model.fit(X_train, y_train)
    t_end = time.time()
    train_time = t_end - t_start

    # 5. 推理
    inf_start = time.time()
    y_pred = model.predict(X_test)
    inf_end = time.time()
    
    # 指标计算
    acc = accuracy_score(y_test, y_pred)
    throughput = len(y_test) / (inf_end - inf_start)
    final_metrics = get_metrics()

    # --- 结果导出为 JSON ---
    benchmark_data = {
        "metadata": {
            "instance_type": INSTANCE_TYPE,
            "timestamp": time.ctime(),
            "dataset_size": len(all_sentiments)
        },
        "performance": {
            "accuracy": round(acc, 4),
            "training_time_sec": round(train_time, 2),
            "inference_throughput_pps": round(throughput, 2),
            "peak_memory_mb": round(final_metrics['memory_usage_mb'], 2),
            "cpu_usage_percent": final_metrics['cpu_usage']
        }
    }

    # 自动生成带时间戳的文件名，防止覆盖
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{INSTANCE_TYPE}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(benchmark_data, f, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    print(f"\nBenchmark finished! Results saved to {output_file}")
    print(f"Accuracy: {acc:.4f} | Throughput: {throughput:.2f} rev/s")

if __name__ == "__main__":
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    main()