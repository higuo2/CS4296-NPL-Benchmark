# CS4296 NLP Cost-Performance Benchmark on AWS EC2

Technical project for CS4296 (Spring 2026): benchmarking a lightweight NLP sentiment pipeline on AWS EC2 to compare cost-performance trade-offs for SME-like CPU workloads.

## Project Goal

This project evaluates which EC2 instance type is most cost-effective for a CPU-bound sentiment analysis workflow.

- Task: binary sentiment classification on IMDB reviews
- Dataset: IMDB 50K Movie Reviews
- Pipeline: NLTK preprocessing + TF-IDF (`max_features=5000`) + Logistic Regression
- Target instances: `t3.micro`, `t3.small`, `t3.medium`
- Repeated runs: `N_TRIALS = 3` per instance

## Repository Structure

- `pipeline_aws.py`: run benchmark on AWS EC2 using data from S3
- `pipeline_local.py`: run benchmark locally using local CSV/ZIP
- `Dockerfile`: containerized runtime for AWS benchmark script
- `requirements.txt`: Python dependencies
- `benchmark_t3.micro.json`: sample benchmark output
- `benchmark_t3.small.json`: sample benchmark output
- `benchmark_t3.medium.json`: sample benchmark output

## Requirements

- Python 3.9+ (tested on Ubuntu in AWS Academy lab)
- Access to AWS EC2 and S3
- IAM instance profile/role that can read your S3 object

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset Setup

### AWS mode (`pipeline_aws.py`)

1. Create/upload to S3:
   - Bucket: your choice (for example: `cs4296-nlp-bench-higuo-20260420`)
   - Object key: **`IMDB Dataset.csv`** (must match exactly)
2. Attach an IAM role/profile to EC2 that can read this object.
3. Set environment variable before running:

```bash
export S3_BUCKET=your-bucket-name
```

### Local mode (`pipeline_local.py`)

Place one of the following in project root:

- `IMDB Dataset.csv`, or
- `IMDB Dataset.csv.zip` (script auto-extracts)

## How To Run

### A) Run locally

```bash
python pipeline_local.py
```

Output:

- `benchmark_local-cpu.json`

### B) Run on EC2 (AWS benchmark)

```bash
export S3_BUCKET=your-bucket-name
python pipeline_aws.py
```

Output (depends on detected instance type):

- `benchmark_t3.micro.json`
- `benchmark_t3.small.json`
- `benchmark_t3.medium.json`

## Benchmark Procedure for Proposal

To match proposal requirements:

1. Run `pipeline_aws.py` on `t3.micro`
2. Run `pipeline_aws.py` on `t3.small`
3. Run `pipeline_aws.py` on `t3.medium`
4. Keep the three generated JSON files for comparison/reporting

Each script execution already performs 3 trials and stores mean/std values.

## Docker Usage

Build image:

```bash
docker build -t cs4296-nlp-benchmark .
```

Run image:

```bash
docker run --rm -e S3_BUCKET=your-bucket-name cs4296-nlp-benchmark
```

Default container entrypoint runs `pipeline_aws.py`.

## Output Format

Each benchmark JSON includes:

- `metadata`: instance type, timestamp, trial count
- `performance`: mean metrics across trials
- `variability`: standard deviation across trials
- `cost_analysis`: hourly rate, estimated run cost, throughput per dollar

Main performance metrics:

- `accuracy`
- `training_time_sec`
- `inference_time_sec`
- `throughput_pps`
- `avg_inference_latency_ms`
- `peak_memory_mb`
- `avg_cpu_percent`

## Notes and Limitations

- Pricing values are currently fixed in `pipeline_aws.py` for `us-east-1` on-demand rates.
- Throughput-per-dollar can be very large when inference time and estimated run cost are both very small.
- For report readability, also present practical metrics such as total time per full dataset run.

## Common Issues

- `AccessDenied` from S3:
  - Check EC2 IAM role/profile permissions
  - Confirm bucket name and object key are correct
- `aws: command not found` on EC2:
  - Install CLI (for Ubuntu): `sudo snap install aws-cli --classic`
- `requirements.txt not found`:
  - You are likely not in the project directory; `cd ~/CS4296-NPL-Benchmark`

## Citation / Course Context

- Course: CS4296 Cloud Computing
- Project type: Technical
- Topic: Cost-performance benchmarking of lightweight NLP pipelines on AWS EC2
