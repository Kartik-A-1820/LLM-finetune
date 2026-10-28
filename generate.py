"""Generate one completion per held-out instruction, with identical decoding."""
import argparse
import json
from pathlib import Path
from prepare_data import verify_splits
from train import prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=Path, default=Path('splits'))
    parser.add_argument('--model', default='EleutherAI/pythia-410m')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    args = parser.parse_args()
    rows = verify_splits(args.splits)['test']
    if args.adapter and (args.adapter / 'split_manifest.json').read_bytes() != (args.splits / 'manifest.json').read_bytes():
        raise ValueError('Adapter was not trained using this split manifest')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for row in rows:
            inputs = tokenizer(prompt(row['instruction']), return_tensors='pt').to(model.device)
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(result[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            handle.write(json.dumps({'id': row['id'], 'completion': text, 'model': args.model, 'adapter': str(args.adapter) if args.adapter else None, 'max_new_tokens': args.max_new_tokens}) + '\n')
if __name__ == '__main__':
    main()
