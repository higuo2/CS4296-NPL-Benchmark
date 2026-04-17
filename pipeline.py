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
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 全自动配置区 ---
BUCKET_NAME = "你的S3桶名"  # 请替换为你的真实 S3 Bucket 名称
FILE_KEY = "IMDB Dataset.csv"

def detect_environment():
    """
    全自动环境识别：
    尝试访问 AWS 元数据服务，成功则为 AWS 环境，失败则为本地环境。
    """
    try:
        # IMDSv1 接口，超时设为 1 秒以防本地阻塞
        response = requests.get("http://169.254.169.254/latest/meta-data/instance-type", timeout=1)
        return response.text, True
    except:
        return "local-cpu", False

# 初始化环境参数
INSTANCE_TYPE, IS_AWS_ENV = detect_environment()
USE_S3 = IS_AWS_ENV  # 仅在 AWS 环境下默认启用 S3

# --- 监控函数 ---
def get_metrics():
    """获取当前的 CPU 使用率和内存占用 (MB)"""
    process = psutil.Process(os.getpid())
    return {
        "cpu_usage": psutil.cpu_percent(interval=0.1),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024)
    }

# --- 增强型全自动数据加载 ---
def load_data():
    if USE_S3:
        try:
            print(f"AWS Environment Detected ({INSTANCE_TYPE}). Connecting to S3: {BUCKET_NAME}...")
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
            # 使用流式读取，避免 1GB 内存溢出
            return pd.read_csv(response['Body'], chunksize=5000)
        except Exception as e:
            print(f"S3 Access Error: {e}. Falling back to local file...")
            return pd.read_csv(FILE_KEY, chunksize=5000)
    else:
        print(f"Local Environment Detected. Loading from local disk: {FILE_KEY}...")
        if not os.path.exists(FILE_KEY):
            raise FileNotFoundError(f"Error: Dataset {FILE_KEY} not found in current directory.")
        return pd.read_csv(FILE_KEY, chunksize=5000)

# --- NLP 清洗函数 ---
def clean_text(text):
    if not isinstance(text, str): return ""
    # 转换为小写并移除 HTML 标签及特殊字符
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # 去停用词与词干提取
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def main():
    print(f"=== Starting Benchmark on: {INSTANCE_TYPE} ===")
    
    all_cleaned_reviews = []
    all_sentiments = []
    
    # 1. 加载与分块清洗 (内存优化核心)
    try:
        reader = load_data()
        for i, chunk in enumerate(reader):
            chunk['review'] = chunk['review'].apply(clean_text)
            all_cleaned_reviews.extend(chunk['review'].tolist())
            all_sentiments.extend(chunk['sentiment'].tolist())
            print(f"Chunk {i+1} processed. Peak Memory: {get_metrics()['memory_usage_mb']:.2f} MB")
            gc.collect() # 显式内存回收
    except Exception as e:
        print(f"Execution failed: {e}")
        return

    # 2. 特征提取 (TF-IDF)
    print("Vectorizing text data...")
    tfidf = TfidfVectorizer(max_features=5000, dtype=np.float32)
    X = tfidf.fit_transform(all_cleaned_reviews)
    y = [1 if s == 'positive' else 0 for s in all_sentiments]
    
    # 释放原始文本内存
    del all_cleaned_reviews
    gc.collect()

    # 3. 数据集拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 模型训练 (Benchmark 重点)
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    t_start = time.time()
    model.fit(X_train, y_train)
    t_end = time.time()
    train_time = t_end - t_start

    # 5. 推理吞吐量测试
    inf_start = time.time()
    y_pred = model.predict(X_test)
    inf_end = time.time()
    
    # 6. 指标统计
    acc = accuracy_score(y_test, y_pred)
    throughput = len(y_test) / (inf_end - inf_start)
    final_metrics = get_metrics()

    # 7. 导出 JSON 结果
    benchmark_data = {
        "metadata": {
            "instance_type": INSTANCE_TYPE,
            "timestamp": time.ctime(),
            "dataset_size": len(all_sentiments),
            "storage_source": "S3" if USE_S3 else "Local"
        },
        "performance": {
            "accuracy": round(acc, 4),
            "training_time_sec": round(train_time, 2),
            "inference_throughput_pps": round(throughput, 2),
            "peak_memory_mb": round(final_metrics['memory_usage_mb'], 2),
            "cpu_usage_percent": final_metrics['cpu_usage']
        }
    }

    # 自动生成文件名 (带时间戳避免覆盖)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{INSTANCE_TYPE}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(benchmark_data, f, indent=4)
    
    print("-" * 30)
    print(f"Benchmark Results for {INSTANCE_TYPE}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Peak Memory: {final_metrics['memory_usage_mb']:.2f} MB")
    print(f"Results saved to: {output_file}")
    print("-" * 30)

if __name__ == "__main__":
    # NLTK 资源预下载
    for res in ['stopwords', 'punkt', 'wordnet']:
        nltk.download(res, quiet=True)
    main()