"""Deterministic, instruction-grouped train/validation/test splits."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

def instruction_id(text):
    normalized = re.sub(r'\s+', ' ', text).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def split_data(paths, output, seed=42):
    groups = {}
    for path in paths:
        for row in read_jsonl(path):
            if not isinstance(row.get('instruction'), str) or not row['instruction'].strip() or not isinstance(row.get('output'), str) or not row['output'].strip():
                raise ValueError('Each row requires nonempty instruction and output strings')
            key = instruction_id(row['instruction'])
            # One deterministic representative per exact normalized instruction.
            groups.setdefault(key, {'id': key, 'instruction': row['instruction'], 'output': row['output'], **({'tests': row['tests']} if row.get('tests') else {})})
    keys = sorted(groups)
    if len(keys) < 10:
        raise ValueError('At least 10 distinct instructions are required')
    random.Random(seed).shuffle(keys)
    n_test = max(1, int(len(keys) * .1))
    n_val = max(1, int(len(keys) * .1))
    parts = {'test': keys[:n_test], 'validation': keys[n_test:n_test+n_val], 'train': keys[n_test+n_val:]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': seed, 'grouping': 'exact normalized instruction; no fuzzy deduplication', 'sources': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}, 'splits': {}}
    for name, ids in parts.items():
        target = output / f'{name}.jsonl'
        target.write_text(''.join(json.dumps(groups[key], ensure_ascii=False) + '\n' for key in ids), encoding='utf-8')
        manifest['splits'][name] = {'count': len(ids), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

def verify_splits(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    result, seen = {}, set()
    for name in ('train', 'validation', 'test'):
        path = directory / f'{name}.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['splits'][name]['sha256']:
            raise ValueError(f'{name} split changed since preparation')
        rows = read_jsonl(path)
        ids = [instruction_id(row['instruction']) for row in rows]
        if len(rows) != manifest['splits'][name]['count'] or len(set(ids)) != len(ids) or any(row['id'] != key for row, key in zip(rows, ids)):
            raise ValueError(f'{name} split identity mismatch')
        if seen.intersection(ids):
            raise ValueError('Instruction leakage across splits')
        seen.update(ids)
        result[name] = rows
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('splits'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(split_data(args.sources, args.output, args.seed), indent=2))
