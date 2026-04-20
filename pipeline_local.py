import pandas as pd
import numpy as np
import re
import sys
import time
import psutil
import os

# Windows 控制台默认 GBK，直接 print emoji 会 UnicodeEncodeError
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import nltk
import gc
import json
import threading
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# ⚙️ 本地环境配置
# ==========================================
LOCAL_FILE = "IMDB Dataset.csv"  # 需与脚本同目录
LOCAL_ZIP = "IMDB Dataset.csv.zip"
N_TRIALS = 3                     # Proposal 要求：至少3次运行取平均

# ==========================================
# 📊 资源监控 (后台线程)
# ==========================================
class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_memory_mb = 0
        self.cpu_samples = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _monitor(self):
        proc = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            self.peak_memory_mb = max(self.peak_memory_mb, mem_mb)
            self.cpu_samples.append(psutil.cpu_percent(interval=0))
            time.sleep(self.interval)

# ==========================================
# 🌍 环境识别 & 数据加载 (本地)
# ==========================================
def detect_environment():
    return "local-cpu", False

def load_data():
    print(f"[*] Local mode: Loading {LOCAL_FILE}")
    if not os.path.exists(LOCAL_FILE):
        if os.path.exists(LOCAL_ZIP):
            print(f"[*] Extracting {LOCAL_ZIP} ...")
            with zipfile.ZipFile(LOCAL_ZIP, "r") as zf:
                zf.extractall(".")
        else:
            raise FileNotFoundError(
                f"Dataset {LOCAL_FILE} not found (and no {LOCAL_ZIP}). "
                "Place the CSV or the zip from the repo in the same directory as this script."
            )
    return pd.read_csv(LOCAL_FILE, chunksize=5000)

# ==========================================
# 🧹 NLP 预处理 (完全一致)
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
# 🏃 单次 Benchmark 逻辑 (完全一致)
# ==========================================
def run_single_trial(trial_id, monitor):
    print(f"\n🔄 Trial {trial_id+1}/{N_TRIALS} started...")
    monitor.peak_memory_mb = 0
    reviews, sentiments = [], []
    
    for chunk in load_data():
        reviews.extend(chunk['review'].apply(clean_text).tolist())
        sentiments.extend(chunk['sentiment'].tolist())
        gc.collect()

    print("[*] Vectorizing (TF-IDF max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000, dtype=np.float32)
    X = tfidf.fit_transform(reviews)
    y = [1 if s == 'positive' else 0 for s in sentiments]
    del reviews, sentiments; gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[*] Training LogisticRegression...")
    monitor.start()
    t0 = time.perf_counter()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    monitor.stop()

    print("[*] Inference testing...")
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = max(time.perf_counter() - t1, 1e-9)
    acc = accuracy_score(y_test, y_pred)

    n_test = len(y_test)
    throughput = n_test / inf_time
    avg_latency_ms = (inf_time / n_test) * 1000
    avg_cpu = np.mean(monitor.cpu_samples) if monitor.cpu_samples else 0
    peak_mem = max(monitor.peak_memory_mb, psutil.Process().memory_info().rss / (1024*1024))

    return {
        "accuracy": round(acc, 4),
        "training_time_sec": round(train_time, 3),
        "inference_time_sec": round(inf_time, 3),
        "throughput_pps": round(throughput, 2),
        "avg_inference_latency_ms": round(avg_latency_ms, 4),
        "peak_memory_mb": round(peak_mem, 2),
        "avg_cpu_percent": round(avg_cpu, 2)
    }

# ==========================================
# 📤 主程序 & 结果固化
# ==========================================
def main():
    for pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
        nltk.download(pkg, quiet=True)

    instance_type, is_aws = detect_environment()
    print(f"\n{'='*45}")
    print(f"  🚀 BENCHMARK START: {instance_type} {'(AWS)' if is_aws else '(Local)'}")
    print(f"{'='*45}")

    trials = []
    for i in range(N_TRIALS):
        monitor = ResourceMonitor(interval=0.5)
        trials.append(run_single_trial(i, monitor))

    agg = {k: {"mean": np.mean([t[k] for t in trials]), "std": np.std([t[k] for t in trials])} for k in trials[0].keys()}

    final_results = {
        "metadata": {"instance_type": instance_type, "is_aws": is_aws, "trials": N_TRIALS, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "performance": {k: v["mean"] for k, v in agg.items()},
        "variability": {k: v["std"] for k, v in agg.items()},
        "cost_analysis": {
            "hourly_rate_usd": 0,
            "cost_per_run_usd": 0,
            "throughput_per_dollar": 0,
            "note": "Local environment: cost metrics disabled."
        }
    }

    out_path = f"benchmark_{instance_type}.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*45}")
    print(f"✅ DONE | Results saved to: {out_path}")
    print(f"📊 Accuracy: {final_results['performance']['accuracy']:.4f}")
    print(f"⏱️  Training: {final_results['performance']['training_time_sec']:.2f}s")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()