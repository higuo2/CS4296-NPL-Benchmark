#!/bin/bash
echo "Starting 3-Trial Structured Benchmark..."

for i in {1..3}
do
   echo "Running trial $i..."
   # 直接运行，不再生成 .log 文件
   python pipeline.py
   # 稍微停顿，确保时间戳文件名不冲突
   sleep 2
done

echo "Done! You should see 3 JSON files in the folder."