from llama_cpp import Llama
import time
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import re

MODEL_PATH = "/home/jiseung/Downloads/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"
llm = None

# gsm8k = load_dataset("gsm8k", "main")["test"]
try:
    results = []
    llm = Llama(model_path = MODEL_PATH, n_ctx=1024, n_threads=4, n_gpu_layers=0,verbose=False)

    asdiv = load_dataset("yimingzhang/asdiv", trust_remote_code = True)["test"]
    def extract_number(text):
        match = re.search(r"###\s*\[?([-+]?\d*\.?\d+)\]?", text)
        if match:
            return float(match.group(1))

        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            return float(numbers[-1])
        return None

    N = 10
    # N = len(gsm8k)

    for idx in range(N):
        # sample = gsm8k[idx]
        sample = asdiv[idx]
        question = sample["text"]
        gold_answer = sample["label"]
        prompt = f"""
    {question}\nAnswer:
    """
        input_tokens = llm.tokenize(prompt.encode("utf-8"))
        input_token_len = len(input_tokens)

        start_time = time.perf_counter()
        output = llm(prompt, max_tokens=128, temperature=0.0)
        end_time = time.perf_counter()

        latency = end_time - start_time

        text_output = output['choices'][0]['text']
        output_tokens = llm.tokenize(text_output.encode("utf-8"))
        output_token_len = len(output_tokens)

        pred_answer = extract_number(text_output)
        gold_number = extract_number(gold_answer)
        is_correct = pred_answer == gold_number

        results.append({
            "input_token_len": input_token_len,
            "output_token_len": output_token_len,
            "latency": latency,
            "is_correct": is_correct
        })

        print(f"Done! {idx+1}/{N} done, latency={latency:.2f}s, input_tokens={input_token_len}, output_tokens={output_token_len}, correct={is_correct}")
        print(f"\n--- Question {idx+1} ---")
        print(question)
        print(f"\n--- Model Output {idx+1} ---")
        print(text_output)
        print(f"---Ground Truth Answer {idx+1} (extracted): {gold_number}")
        print(f"--- Predicted Answer {idx+1} (extracted): {pred_answer}")
        print("-" *30)

    output_lens = np.array([r['output_token_len'] for r in results])
    latencies = np.array([r['latency'] for r in results])
    accuracies = np.array([r['is_correct'] for r in results])

    corr = np.corrcoef(output_lens, latencies)[0,1]
    accuracy_percent = np.mean(accuracies) *100

    print(f"\n output token length vs latency correlation: {corr:.3f}")
    print(f" overall accuracy: {accuracy_percent:.1f}%")

    plt.figure(figsize=(8,6))
    plt.scatter(output_lens, latencies, alpha=0.7, c=accuracies, cmap='coolwarm')
    plt.xlabel("Output Token Length")
    plt.ylabel("Latency (seconds)")
    plt.title(f"Output Tokens vs Latency (corr={corr:.3f}, acc={accuracy_percent:.1f}%)")
    plt.colorbar(label="Correctness (1=correct, 0=incorrect)")
    plt.grid(True, alpha=0.3)
    plt.show()


    print(f"\n=== Output ===\n{output['choices'][0]['text']}")
    print(f"\n=== Latency: {latency:.3f} seconds ===")
finally:
    if llm is not None:
        llm.close()

