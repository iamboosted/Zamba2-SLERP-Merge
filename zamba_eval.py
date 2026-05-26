import torch, json, re, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.modeling_utils

# Patch: skip the strict tie_weights validation that breaks Zamba2
_original_post_init = transformers.modeling_utils.PreTrainedModel.post_init
def _patched_post_init(self):
    try:
        _original_post_init(self)
    except ValueError as e:
        if "tie_weights_keys" in str(e):
            print(f"  [PATCH] Skipping tie_weights validation: known Zamba2 compat issue")
            self.init_weights()
        else:
            raise
transformers.modeling_utils.PreTrainedModel.post_init = _patched_post_init

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

MODELS = {
    "zamba2_base":     "Zyphra/Zamba2-7B",
    "zamba2_instruct": "Zyphra/Zamba2-7B-Instruct",
    "zamba2_merged":   "/workspace/zamba2-7b-merged",
}

PROBLEMS = [
    {"q": "What is 247 * 38? Reply with just the number.", "a": "9386"},
    {"q": "What is 1729 + 4856? Reply with just the number.", "a": "6585"},
    {"q": "What is 15% of 840? Reply with just the number.", "a": "126"},
    {"q": "Solve for x: 5x - 13 = 42. Reply with just the number.", "a": "11"},
    {"q": "Solve for x: 2x + 3 = 4x - 7. Reply with just the number.", "a": "5"},
    {"q": "If f(x) = 3x^2 - 2x + 1, what is f(4)? Reply with just the number.", "a": "41"},
    {"q": "A store sells apples for $1.50 each. If I buy 7 apples and pay with a $20 bill, how much change do I get? Reply with just the number.", "a": "9.50"},
    {"q": "A car uses 8 liters of fuel per 100 km. How many liters does it need for a 350 km trip? Reply with just the number.", "a": "28"},
    {"q": "If 4 workers can build a wall in 6 days, how many days would it take 3 workers? Reply with just the number.", "a": "8"},
    {"q": "A rectangle has a perimeter of 36 cm and a length of 12 cm. What is its width? Reply with just the number.", "a": "6"},
    {"q": "What is the next number in the sequence: 2, 6, 18, 54, ...? Reply with just the number.", "a": "162"},
    {"q": "How many prime numbers are there between 1 and 20? Reply with just the number.", "a": "8"},
    {"q": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Reply with just yes or no.", "a": "yes"},
    {"q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Reply with just the number.", "a": "5"},
    {"q": "What is the sum of the first 10 positive integers? Reply with just the number.", "a": "55"},
    {"q": "What is 17 squared minus 13 squared? Reply with just the number.", "a": "120"},
    {"q": "A clock shows 3:15. What is the angle between the hour and minute hands in degrees? Reply with just the number.", "a": "7.5"},
    {"q": "How many diagonals does a hexagon have? Reply with just the number.", "a": "9"},
    {"q": "If you flip a coin 3 times, how many possible outcomes are there? Reply with just the number.", "a": "8"},
    {"q": "What is the greatest common divisor of 48 and 36? Reply with just the number.", "a": "12"},
]

def extract_answer(text, expected):
    text_lower = text.lower().strip()
    if expected.lower() in ("yes", "no"):
        if "yes" in text_lower and "no" not in text_lower.replace("not", "").replace("know", ""):
            return "yes"
        elif "no" in text_lower:
            return "no"
        for word in text_lower.split():
            if word.strip(".,!") in ("yes", "no"):
                return word.strip(".,!")
        return None
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if not numbers:
        return None
    for n in numbers:
        try:
            if abs(float(n) - float(expected)) < 0.01:
                return n
        except ValueError:
            continue
    return numbers[-1] if numbers else None

results = {}
for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", dtype=torch.bfloat16,
        quantization_config=bnb_config, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    correct = 0
    total = len(PROBLEMS)
    details = []
    for i, prob in enumerate(PROBLEMS):
        inputs = tokenizer(prob["q"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.0, do_sample=False,
                                 repetition_penalty=1.1)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_text = response[len(prob["q"]):].strip()
        found = extract_answer(gen_text, prob["a"])
        is_correct = found is not None and (
            (prob["a"].lower() in ("yes","no") and found.lower() == prob["a"].lower()) or
            (prob["a"].lower() not in ("yes","no") and abs(float(found) - float(prob["a"])) < 0.01)
        ) if found else False
        mark = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1
        print(f"  {mark} Q{i+1}: expected={prob['a']}, got={found or '???'} | {gen_text[:80]}")
        details.append({"q": prob["q"], "expected": prob["a"], "got": found, "correct": is_correct, "raw": gen_text[:200]})
    score = correct / total * 100
    results[model_name] = {"score": score, "correct": correct, "total": total, "details": details}
    print(f"\n  SCORE: {correct}/{total} ({score:.0f}%)")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"  FINAL COMPARISON")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name:20s}: {r['correct']}/{r['total']} ({r['score']:.0f}%)")

with open("/workspace/zamba_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to /workspace/zamba_eval_results.json")
