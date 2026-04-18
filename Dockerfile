FROM python:3.9-slim

# 设置 Python 环境变量：不生成 .pyc 文件，且让日志直接输出到终端
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 先安装系统依赖（如果以后需要处理更复杂的 C 编译库，这一行很有用）
# 对于你现在的依赖，slim 版的基础镜像已经足够

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 NLTK 数据
# 增加 punkt_tab 以适配最新的 NLTK 3.8.1+ 版本
RUN python -m nltk.downloader stopwords punkt punkt_tab wordnet

# 复制项目代码（注意：如果你的代码文件名是 pipe_aws.py，这里要对齐）
COPY pipe_aws.py ./pipeline.py

# 设置默认 S3 桶名
ENV S3_BUCKET=your-imdb-bucket-name

# 运行入口
CMD ["python", "pipeline.py"]