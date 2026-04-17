FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 预先下载 NLTK 数据，减少运行时的网络开销
RUN python -m nltk.downloader stopwords punkt wordnet
COPY . .
CMD ["python", "pipeline.py"]