from llama_cpp import Llama
import time
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import re
from sklearn.linear_model import LinearRegression
import json

MODEL_PATH = "/home/jiseung/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
llm = None

try:
    llm = Llama(model_path = MODEL_PATH, n_ctx=4096, n_threads=4, n_gpu_layers=0,verbose=False)

    gsm8k = load_dataset("gsm8k", "main")["test"]

    results = []

    def extract_number(text):
        match = re.search(r"###\s*\[?([-+]?\d*\.?\d+)\]?", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        lines = text.strip().split("\n")
        for line in reversed(lines[-3:]):
            numbers = re.findall(r"[-+]?\d*\.?\d+", line)
            if numbers:
                try:
                    return float(numbers[-1])
                except ValueError:
                    pass
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    N = 30
    # N = len(gsm8k)

    for idx in range(N):
        sample = gsm8k[idx]
        question = sample["question"]
        gold_answer = sample["answer"]
        prompt = f"""
    Lets's solve this math problem step by step.

    {question}

    Please solve the problem step by step, showing your reasoning.
    At the end, write the final answer after ### like this:
    ### 42

    Let's think step by step.
    """
        input_tokens = llm.tokenize(prompt.encode("utf-8"))
        input_token_len = len(input_tokens)

        start_time = time.perf_counter()
        output = llm(prompt, max_tokens=4096, temperature=0.0, stop = ["\nQ:", "\nQ", "\nA:"])
        end_time = time.perf_counter()

        latency = end_time - start_time

        text_output = output['choices'][0]['text']
        output_tokens = llm.tokenize(text_output.encode("utf-8"))
        output_token_len = len(output_tokens)

        pred_answer = extract_number(text_output)
        gold_number = extract_number(gold_answer)
        is_correct = pred_answer == gold_number

        total_tokens = input_token_len + output_token_len
        tokens_per_second = total_tokens / latency if latency > 0 else 0

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "gold_extracted": gold_number,
            "predicted_answer": pred_answer,
            "full_model_output:": text_output,
            "input_token_len": input_token_len,
            "output_token_len": output_token_len,
            "latency": latency,
            "tokens_per_second": tokens_per_second,
            "is_correct": is_correct
        })

        print(f"Done! {idx+1}/{N} done, latency={latency:.2f}s, input_tokens={input_token_len}, output_tokens={output_token_len}, correct={is_correct}")
        print(f"\n--- Question {idx+1} ---")
        print(question)
        print(f"\n--- Model Output {idx+1} ---")
        print(text_output)
        print(f"--- Ground Truth Answer {idx+1} (extracted): {gold_number}")
        print(f"--- Predicted Answer {idx+1} (extracted): {pred_answer}")
        print("-" *30)

    output_lens = np.array([r['output_token_len'] for r in results])
    input_lens = np.array([[r['input_token_len'] for r in results])
    latencies = np.array([r['latency'] for r in results])
    accuracies = np.array([r['is_correct'] for r in results])

    corr = np.corrcoef(output_lens, latencies)[0,1]
    accuracy_percent = np.mean(accuracies) *100

    print(f"\n output token length vs latency correlation: {corr:.3f}")
    print(f" overall accuracy: {accuracy_percent:.1f}%")

    plt.figure(figsize=(8,6))
    plt.scatter(output_lens, latencies, alpha=0.7)
    plt.xlabel("Output Token Length")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency vs Output Token Length")
    plt.grid(True, alpha=0.3)
    plt.savefig("latency_vs_output_tokens.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(input_lens, latencies, alpha=0.7)
    plt.xlabel("Input Token Length")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency vs Input Token Length")
    plt.grid(True, alpha=0.3)
    plt.savefig("latency_vs_input_tokens.png", dpi=300)
    plt.show()

    X = np.column_stack((output_lens, input_lens))
    y = latencies
    reg = LinearRegression()
    reg.fit(X,y)

    a = reg.coef_[0]
    b = reg.coef_[1]
    c = reg.intercept_

    print("\nLinear Regression Output:")
    print(f"latency = {a:.6f} * (output tokens) + {b:.6f} * (input tokens) + {c:.6f}")

    


    with open("inference_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)                      

    print(f"\n=== Output ===\n{output['choices'][0]['text']}")
    print(f"\n=== Latency: {latency:.3f} seconds ===")
finally:
    if llm is not None:
        llm.close()

