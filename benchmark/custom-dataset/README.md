# Run benchmark

The dataset is a jsonl file with the following format:
```json
{"prompt": "Write a short story about a robot who learns to love."}
{"prompt": "Summarize the main differences between TCP and UDP."}
{"prompt": "Generate a SQL query to find all users who signed up in the last 30 days."}
{"prompt": "Translate the following sentence into French: 'The quick brown fox jumps over the lazy dog.'"}
{"prompt": "Explain how gradient descent works in machine learning."}
```

## Run benchmark

```bash
DATASET_PATH=./sample.jsonl

python3 -m sglang.bench_serving \
	    --backend sglang \
		--num-prompts 100 \
		--random-output-len 100 \
		--host 127.0.0.1 \
		--port 8188 \
		--max-concurrency 16 \
		--dataset-name custom-dataset \
		--dataset-path $DATASET_PATH \
		--request-rate 2.5
```