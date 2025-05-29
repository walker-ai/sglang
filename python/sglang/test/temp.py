from openai import OpenAI
import time
import random

# Modify OpenAI's API key and API base to use vLLM's API server.
cnt = 0
prompt = input('prompt: ')
ip_list = ["33.180.3.251", "33.180.25.9", "33.180.24.136", "33.180.3.99", "33.180.27.109", "33.180.25.18", "33.180.24.234", "33.180.3.93", "33.180.27.240", "33.180.24.114", "33.180.24.191", "33.180.3.52", "33.180.3.6", "33.180.3.220", "33.180.24.70", "33.180.25.121", "33.180.25.20", "33.180.24.62", "33.180.24.27", "33.180.3.184", "33.180.24.156", "33.180.3.211", "33.180.24.9", "33.180.27.171", "33.180.25.140", "33.180.24.176", "33.180.25.45", "33.180.24.50", "33.180.24.26", "33.180.24.209", "33.180.3.146", "33.180.25.128", "33.180.3.185", "33.180.24.134", "33.180.3.241", "33.180.3.216", "33.180.24.37", "33.180.3.82", "33.180.24.66", "33.180.3.88", "33.180.25.142", "33.180.25.39", "33.180.3.114", "33.180.27.164", "33.180.24.147", "33.180.24.51", "33.180.27.198", "33.180.24.190", "33.180.3.233", "33.180.24.251", "33.180.3.207", "33.180.24.125", "33.180.25.134", "33.180.3.175", "33.180.3.205", "33.180.24.235", "33.180.3.177", "33.180.24.98", "33.180.3.240", "33.180.24.219", "33.180.3.48", "33.180.25.40", "33.180.24.165", "33.180.3.130", "33.180.3.22", "33.180.25.123", "33.180.3.168", "33.180.25.124", "33.180.25.29", "33.180.24.247", "33.180.24.29", "33.180.25.137", "33.180.27.135", "33.180.25.4", "33.180.24.146", "33.180.27.210", "33.180.3.73", "33.180.3.24", "33.180.25.73", "33.180.3.43", "33.180.3.92", "33.180.25.41", "33.180.3.226", "33.180.3.131", "33.180.27.188", "33.180.25.58"]
ip_list = ["127.0.0.1"]
for ip in ip_list:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{ip}:8188/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    model = 'auto'
    for p_len in range(0, 8000):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": str(random.randint(1, 100000)) + "你好"*p_len
                }
            ],
            stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
            model=model,
            max_tokens=2,
            temperature=0.0,
            extra_body={
                "trace_id": "1234567890"
            }
        )
        tokens = 0
        yield_cnt = 0
        prompt_tokens = 0
        cached_tokens = 0
        start_time = time.time()
        think_label = True
        for ck in chat_completion:
            if ck.choices and ck.choices[0].delta.content:
                print(ck.choices[0].delta.content, flush=True, end='')
            if tokens == 0 and yield_cnt == 0:
                ttft_time = time.time() - start_time
            # tokens += 1
            yield_cnt += 1
            if ck.usage:
                # print(f'kkk = {ck.usage}')
                tokens = ck.usage.completion_tokens
                prompt_tokens = ck.usage.prompt_tokens
                if ck.usage.prompt_tokens_details and ck.usage.prompt_tokens_details.cached_tokens:
                    cached_tokens = ck.usage.prompt_tokens_details.cached_tokens
        end_time = time.time()
        cost_time = end_time - start_time
        print('\n\ninput_tokens=({}/{}), new_tokens={}, tokens_per_yield={:.2f}, latency={:.2f}s, first_token_rt={:.2f}ms, time_per_token={:.2f}ms, tokens_per_sec={:.2f}'.format(cached_tokens, prompt_tokens, tokens, tokens/yield_cnt, cost_time, ttft_time*1000, (cost_time-ttft_time)*1000/(tokens-1), (tokens-1)/(cost_time-ttft_time)))
