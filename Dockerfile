FROM python:3.9-slim

# Keep Python output unbuffered for live benchmark logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first to maximize Docker layer cache hits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK assets used by pipeline_aws.py / pipeline_local.py.
RUN python -m nltk.downloader stopwords punkt punkt_tab wordnet omw-1.4

# Copy project scripts (use actual filenames in the repository).
COPY pipeline_aws.py pipeline_local.py ./

# Runtime configuration: override at `docker run -e S3_BUCKET=...`.
ENV S3_BUCKET=your-imdb-bucket-name

# Default entrypoint runs the AWS benchmark pipeline.
CMD ["python", "pipeline_aws.py"]