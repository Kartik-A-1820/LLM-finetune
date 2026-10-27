"""Supported response-only LoRA training entrypoint; test split never trains."""
import argparse
from pathlib import Path
from prepare_data import verify_splits

def prompt(instruction):
    return f'### Instruction:\n{instruction}\n\n### Response:\n'

def tokenize_example(row, tokenizer, max_length):
    prefix = tokenizer.encode(prompt(row['instruction']), add_special_tokens=False)
    answer = tokenizer.encode(row['output'], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(answer) > max_length:
        raise ValueError('Example exceeds max_length; increase it or curate data explicitly rather than silently truncating code')
    return {'input_ids': prefix + answer, 'attention_mask': [1] * (len(prefix) + len(answer)), 'labels': [-100] * len(prefix) + answer}

class ResponseCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        import torch
        size = max(len(row['input_ids']) for row in batch)
        return {key: torch.tensor([row[key] + [fill] * (size - len(row[key])) for row in batch]) for key, fill in [('input_ids', self.pad_id), ('attention_mask', 0), ('labels', -100)]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=Path, default=Path('splits'))
    parser.add_argument('--model', default='EleutherAI/pythia-410m')
    parser.add_argument('--output', default='outputs/adapter')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--four-bit', action='store_true')
    args = parser.parse_args()
    rows = verify_splits(args.splits)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(42)
    if args.four_bit and not torch.cuda.is_available():
        raise ValueError('--four-bit requires a supported CUDA/bitsandbytes setup')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = {name: [tokenize_example(row, tokenizer, args.max_length) for row in rows[name]] for name in ('train', 'validation')}
    kwargs = {'quantization_config': BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16), 'device_map': {'': torch.cuda.current_device()}} if args.four_bit else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=['query_key_value'], bias='none', task_type='CAUSAL_LM'))
    training = TrainingArguments(output_dir=args.output, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=args.epochs, learning_rate=3e-4, eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, save_total_limit=2, report_to='none', seed=42, fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=training, train_dataset=encoded['train'], eval_dataset=encoded['validation'], data_collator=ResponseCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    # Preserve exact data provenance alongside the adapter.
    (Path(args.output) / 'split_manifest.json').write_bytes((args.splits / 'manifest.json').read_bytes())

if __name__ == '__main__':
    main()
