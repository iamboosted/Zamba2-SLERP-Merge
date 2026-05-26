import json, os
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download
import torch, shutil

MODEL_A = "Zyphra/Zamba2-7B"           # base
MODEL_B = "Zyphra/Zamba2-7B-Instruct"  # instruct
OUT_DIR = "/workspace/zamba2-7b-merged"
NUM_LAYERS = 81

def make_curve(values, n=NUM_LAYERS):
    out = []
    for i in range(n):
        frac = i / (n - 1) * (len(values) - 1)
        lo = int(frac)
        hi = min(lo + 1, len(values) - 1)
        w = frac - lo
        out.append(values[lo] * (1 - w) + values[hi] * w)
    return out

ATTN_CURVE = make_curve([0.3, 0.5, 0.7, 0.7, 0.5, 0.3])
FF_CURVE   = make_curve([0.4, 0.6, 0.8, 0.8, 0.6, 0.4])
MAMBA_T    = 0.75
NORM_T     = 0.5
DEFAULT_T  = 0.5

def get_t(key):
    if not key.startswith("model.layers."):
        return DEFAULT_T
    parts = key.split(".")
    layer_idx = int(parts[2])
    rest = ".".join(parts[3:])
    if "self_attn" in rest:
        return ATTN_CURVE[layer_idx]
    elif "feed_forward" in rest:
        return FF_CURVE[layer_idx]
    elif "mamba" in rest:
        return MAMBA_T
    elif "layernorm" in rest:
        return NORM_T
    elif rest == "linear.weight":
        return DEFAULT_T
    return DEFAULT_T

def slerp(v1, v2, t):
    v1_f = v1.float()
    v2_f = v2.float()
    if v1_f.shape != v2_f.shape or v1_f.dim() < 2 or v1_f.numel() < 16:
        if v1_f.shape != v2_f.shape:
            print(f"    SHAPE MISMATCH: {v1.shape} vs {v2.shape}, using model A")
            return v1
        return torch.lerp(v1_f, v2_f, t).to(v1.dtype)
    orig_shape = v1_f.shape
    v1_2d = v1_f.reshape(orig_shape[0], -1)
    v2_2d = v2_f.reshape(orig_shape[0], -1)
    v1_norm = torch.nn.functional.normalize(v1_2d, dim=1)
    v2_norm = torch.nn.functional.normalize(v2_2d, dim=1)
    dot = (v1_norm * v2_norm).sum(dim=1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(dot)
    near_parallel = (omega.abs() < 1e-6).squeeze()
    sin_omega = torch.sin(omega)
    sin_omega = torch.where(sin_omega.abs() < 1e-8, torch.ones_like(sin_omega), sin_omega)
    coeff1 = torch.sin((1 - t) * omega) / sin_omega
    coeff2 = torch.sin(t * omega) / sin_omega
    result = coeff1 * v1_2d + coeff2 * v2_2d
    if near_parallel.any():
        lerp_result = (1 - t) * v1_2d + t * v2_2d
        result[near_parallel] = lerp_result[near_parallel]
    return result.reshape(orig_shape).to(v1.dtype)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Downloading model A index...")
    idx_a = hf_hub_download(MODEL_A, "model.safetensors.index.json")
    with open(idx_a) as f:
        index_a = json.load(f)
    print("Downloading model B index...")
    idx_b = hf_hub_download(MODEL_B, "model.safetensors.index.json")
    with open(idx_b) as f:
        index_b = json.load(f)
    shards_a = sorted(set(index_a["weight_map"].values()))
    print(f"Model A has {len(shards_a)} shards")
    all_keys_done = set()
    out_index = {"metadata": {"format": "pt"}, "weight_map": {}}
    b_key_to_shard = index_b["weight_map"]
    shard_num = 0
    for shard_file in shards_a:
        shard_num += 1
        out_shard_name = f"model-{shard_num:05d}-of-{len(shards_a):05d}.safetensors"
        print(f"\n--- Shard {shard_num}/{len(shards_a)}: {shard_file} ---")
        path_a = hf_hub_download(MODEL_A, shard_file)
        tensors_a = load_file(path_a)
        needed_b_shards = set()
        for key in tensors_a:
            if key in b_key_to_shard:
                needed_b_shards.add(b_key_to_shard[key])
        tensors_b = {}
        for b_shard in needed_b_shards:
            path_b = hf_hub_download(MODEL_B, b_shard)
            tensors_b.update(load_file(path_b))
        merged = {}
        for key, va in tensors_a.items():
            t = get_t(key)
            if key in tensors_b:
                vb = tensors_b[key]
                merged[key] = slerp(va, vb, t)
                print(f"  SLERP t={t:.3f} {key} {list(va.shape)}")
            else:
                merged[key] = va
                print(f"  COPY (no match in B) {key}")
            out_index["weight_map"][key] = out_shard_name
            all_keys_done.add(key)
        out_path = os.path.join(OUT_DIR, out_shard_name)
        save_file(merged, out_path)
        print(f"  Saved {out_path}")
        del tensors_a, tensors_b, merged
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(out_index, f, indent=2)
    print("\nCopying config and tokenizer from base model...")
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"]:
        try:
            src = hf_hub_download(MODEL_A, fname)
            shutil.copy2(src, os.path.join(OUT_DIR, fname))
            print(f"  Copied {fname}")
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
    print(f"\nDone! Merged model saved to {OUT_DIR}")
    print(f"Total parameters merged: {len(all_keys_done)}")

if __name__ == "__main__":
    main()
