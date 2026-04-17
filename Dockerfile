FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords punkt wordnet omw-1.4
# 将数据和代码都拷贝进去
COPY . . 
CMD ["python", "pipeline.py"]