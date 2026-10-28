import json
import pytest
from prepare_data import split_data, verify_splits
from evaluate import score
from train import tokenize_example, ResponseCollator

def test_splits_are_disjoint_deterministic_and_bound(tmp_path):
    source = tmp_path / 'source.jsonl'
    rows = [{'instruction': f'write function {i}', 'output': 'x=1'} for i in range(30)]
    rows += [{'instruction': '  WRITE   function 0 ', 'output': 'x=2'}]
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    first, second = tmp_path/'a', tmp_path/'b'
    assert split_data([source], first) == split_data([source], second)
    parts = verify_splits(first)
    assert sum(map(len, parts.values())) == 30
    (first/'test.jsonl').write_text('')
    with pytest.raises(ValueError):
        verify_splits(first)

def test_code_scoring_requires_complete_unique_predictions():
    rows = [{'id':'a', 'output':'x=1'}, {'id':'b', 'output':'y=2'}]
    result = score(rows, [{'id':'a','completion':'```python\nx=1\n```'}, {'id':'b','completion':'not valid :'}])
    assert result['syntax_valid_rate'] == .5
    assert result['pass_at_1'] is None
    for predictions in [[], [{'id':'a','completion':'x=1'}]*2]:
        with pytest.raises(ValueError):
            score(rows, predictions)

def test_only_response_and_real_eos_are_supervised():
    class Tokenizer:
        eos_token_id = 9
        def encode(self, text, **kwargs):
            return [1,2] if text.startswith('###') else [3]
    result = tokenize_example({'instruction':'x','output':'y'}, Tokenizer(), 10)
    assert result['labels'] == [-100,-100,3,9]
    with pytest.raises(ValueError):
        tokenize_example({'instruction':'x','output':'y'}, Tokenizer(), 2)
    torch = pytest.importorskip('torch')
    padded = ResponseCollator(9)([result, {'input_ids':[3,9], 'attention_mask':[1,1], 'labels':[3,9]}])
    assert padded['labels'][1].tolist() == [3,9,-100,-100]

@pytest.mark.parametrize('text', ['```', '```python', 'return 1', 'break', ''])
def test_malformed_or_uncompilable_completions_are_invalid(text):
    from evaluate import code_only
    result=score([{'id':'x','output':'x=1'}],[{'id':'x','completion':text}])
    assert result['syntax_valid_rate']==0
    assert isinstance(code_only(text),str)

@pytest.mark.parametrize('code,tests,success', [
    ('x=1','assert x==1',True),
    ('x=1','assert x==2',False),
    ('import sys; sys.exit(0)','assert False',False),
    ('import os; os._exit(0)','assert False',False),
    ('print("noise"); x=1','assert x==1',True),
])
def test_harness_receipt_requires_tests_to_finish(tmp_path,code,tests,success):
    import subprocess,sys
    from evaluate import harness_source
    marker=b'test-receipt'
    script=tmp_path/'harness.py'
    script.write_text(harness_source(code,tests,marker))
    result=subprocess.run([sys.executable,'-I',str(script)],capture_output=True,timeout=5)
    observed=result.returncode==0 and result.stdout==marker+b'\n'
    assert observed is success
