import pandas as pd
import numpy as np
import re
import sys
import time
import psutil
import os

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import nltk
import gc
import json
import boto3
import requests
import threading
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# ⚙️ AWS 环境配置
# ==========================================
# 建议通过环境变量设置桶名，更安全：export S3_BUCKET="your-bucket-name"
S3_BUCKET = os.getenv("S3_BUCKET", "你的S3桶名") 
FILE_KEY = "IMDB Dataset.csv"
N_TRIALS = 3
# AWS 官方定价 (us-east-1 On-Demand)
PRICING_USD_PER_HR = { "t3.micro": 0.0104, "t3.small": 0.0208, "t3.medium": 0.0416 }

# ==========================================
# 📊 资源监控 (全生命周期监控)
# ==========================================
class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_memory_mb = 0
        self.cpu_samples = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()

    def _monitor(self):
        proc = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            try:
                # 捕捉 RSS 内存
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, mem_mb)
                # 捕捉当前进程 CPU 占用
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
            except:
                pass
            time.sleep(self.interval)

# ==========================================
# 🌍 环境识别 (兼容 IMDSv1 & IMDSv2)
# ==========================================
def detect_environment():
    """使用 Token 方式安全获取 EC2 实例元数据"""
    try:
        # 1. 尝试获取 IMDSv2 Token (有效期 60 秒)
        token_url = "http://169.254.169.254/latest/api/token"
        token_resp = requests.put(token_url, headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}, timeout=2)
        token = token_resp.text

        # 2. 使用 Token 获取实例类型
        meta_url = "http://169.254.169.254/latest/meta-data/instance-type"
        resp = requests.get(meta_url, headers={"X-aws-ec2-metadata-token": token}, timeout=2)
        return resp.text.strip(), True
    except Exception:
        # 如果报错，可能是本地环境或 IMDS 不可用
        return "local-cpu", False

def load_data_s3():
    print(f"[*] AWS mode: Streaming from s3://{S3_BUCKET}/{FILE_KEY}")
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=S3_BUCKET, Key=FILE_KEY)
    # 直接返回 TextIOWrapper 供 pandas 流式读取
    return pd.read_csv(obj['Body'], chunksize=5000)

# ==========================================
# 🧹 NLP 预处理
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

# ==========================================
# 🏃 单次 Benchmark 逻辑
# ==========================================
def run_single_trial(trial_id):
    print(f"\n🔄 Trial {trial_id+1}/{N_TRIALS} started...")
    
    # 启动全过程监控，覆盖数据加载、清洗、向量化和训练
    monitor = ResourceMonitor()
    monitor.start()
    
    start_total = time.time()
    
    # 1. 加载与清洗
    reviews, sentiments = [], []
    for chunk in load_data_s3():
        reviews.extend(chunk['review'].apply(clean_text).tolist())
        sentiments.extend(chunk['sentiment'].tolist())
        gc.collect()

    # 2. 特征工程
    print("[*] Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000, dtype=np.float32)
    X = tfidf.fit_transform(reviews)
    y = [1 if s == 'positive' else 0 for s in sentiments]
    del reviews, sentiments; gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 训练测试
    print("[*] Training model...")
    t0 = time.perf_counter()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # 4. 推理测试
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = max(time.perf_counter() - t1, 1e-9)
    
    # 停止监控
    monitor.stop()
    
    acc = accuracy_score(y_test, y_pred)
    n_test = len(y_test)
    
    # 计算指标
    avg_cpu = np.mean(monitor.cpu_samples) if monitor.cpu_samples else 0
    
    return {
        "accuracy": round(acc, 4),
        "training_time_sec": round(train_time, 3),
        "inference_time_sec": round(inf_time, 3),
        "throughput_pps": round(n_test / inf_time, 2),
        "avg_inference_latency_ms": round((inf_time / n_test) * 1000, 4),
        "peak_memory_mb": round(monitor.peak_memory_mb, 2),
        "avg_cpu_percent": round(avg_cpu, 2)
    }

# ==========================================
# 📤 主程序
# ==========================================
def main():
    # 1. 离线/在线兼容的 NLTK 下载
    print("[*] Initializing NLTK resources...")
    for pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
        nltk.download(pkg, quiet=True)

    # 2. 环境识别
    instance_type, is_aws = detect_environment()
    print(f"\n{'='*45}")
    print(f"  🚀 AWS BENCHMARK START: {instance_type}")
    print(f"{'='*45}")

    # 3. 运行试验
    trials = []
    for i in range(N_TRIALS):
        trials.append(run_single_trial(i))

    # 4. 统计分析
    agg = {k: {"mean": np.mean([t[k] for t in trials]), "std": np.std([t[k] for t in trials])} for k in trials[0].keys()}

    # 5. 成本计算
    hourly_rate = PRICING_USD_PER_HR.get(instance_type, 0)
    # 我们计算单次试验的活跃时间成本（训练+推理）
    active_time_hr = (agg["training_time_sec"]["mean"] + agg["inference_time_sec"]["mean"]) / 3600
    cost_per_run = round(hourly_rate * active_time_hr, 6)
    throughput_per_dollar = round(agg["throughput_pps"]["mean"] / cost_per_run if cost_per_run > 0 else 0, 2)

    final_results = {
        "metadata": {
            "instance_type": instance_type,
            "is_aws": is_aws,
            "trials": N_TRIALS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "performance": {k: v["mean"] for k, v in agg.items()},
        "variability": {k: v["std"] for k, v in agg.items()},
        "cost_analysis": {
            "hourly_rate_usd": hourly_rate,
            "cost_per_run_usd": cost_per_run,
            "throughput_per_dollar": throughput_per_dollar
        }
    }

    # 6. 保存
    out_path = f"benchmark_{instance_type}.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*45}")
    print(f"✅ DONE | Results saved to: {out_path}")
    print(f"📊 Accuracy: {final_results['performance']['accuracy']:.4f}")
    print(f"⏱️  Training: {final_results['performance']['training_time_sec']:.2f}s")
    print(f"💾 Peak Memory: {final_results['performance']['peak_memory_mb']:.2f} MB")
    print(f"💸 Cost/Run: ${cost_per_run:.6f}")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()