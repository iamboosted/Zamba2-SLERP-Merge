from huggingface_hub import hf_hub_download
import json

sep = '=' * 60

for name, model_id in [('Zamba2-7B', 'Zyphra/Zamba2-7B'), ('Zamba2-7B-Instruct', 'Zyphra/Zamba2-7B-Instruct')]:
    print()
    print(sep)
    print(f'  {name}')
    print(sep)
    idx = hf_hub_download(model_id, 'model.safetensors.index.json')
    with open(idx) as f:
        keys = list(json.load(f)['weight_map'].keys())
    
    layer_nums = sorted(set(int(k.split('.')[2]) for k in keys if k.startswith('model.layers.')))
    print(f'Total keys: {len(keys)}')
    print(f'Layer range: {min(layer_nums)} to {max(layer_nums)} ({len(layer_nums)} layers)')
    
    patterns = sorted(set('.'.join(k.split('.')[:2] + k.split('.')[3:]) for k in keys if k.startswith('model.layers.')))
    print('\nParameter patterns per layer:')
    for p in patterns:
        print(f'  {p}')
    
    print('\nNon-layer keys:')
    for k in sorted(keys):
        if not k.startswith('model.layers.'):
            print(f'  {k}')

keys_a, keys_b = [], []
for mid in ['Zyphra/Zamba2-7B', 'Zyphra/Zamba2-7B-Instruct']:
    idx = hf_hub_download(mid, 'model.safetensors.index.json')
    with open(idx) as f:
        k = sorted(json.load(f)['weight_map'].keys())
    if not keys_a: keys_a = k
    else: keys_b = k

if keys_a == keys_b:
    print(f'\nMATCH: Both models have identical structure ({len(keys_a)} keys)')
else:
    print('\nMISMATCH!')
    only_a = set(keys_a) - set(keys_b)
    only_b = set(keys_b) - set(keys_a)
    if only_a: print(f'  Only in Base: {only_a}')
    if only_b: print(f'  Only in Instruct: {only_b}')
