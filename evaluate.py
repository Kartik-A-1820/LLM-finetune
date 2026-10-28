"""Static code metrics and optional isolated, test-backed pass@1."""
import argparse
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from prepare_data import read_jsonl, verify_splits

def code_only(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```') and '\n' in text:
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return text

def harness_source(code, tests, marker):
    # Save harness operations before the completion can change its namespace.
    # Suppress completion output; only the harness writes the success receipt.
    return (
        "import os\n"
        "_write, _exit = os.write, os._exit\n"
        "_stdout = os.dup(1)\n"
        "_null = os.open(os.devnull, os.O_WRONLY)\n"
        "os.dup2(_null, 1)\nos.dup2(_null, 2)\n"
        "_exec, _compile = exec, compile\n"
        "_namespace = {'__name__': '__completion__'}\n"
        "try:\n"
        f"    _exec(_compile({code!r}, '<completion>', 'exec'), _namespace)\n"
        f"    _exec(_compile({tests!r}, '<tests>', 'exec'), _namespace)\n"
        "except BaseException:\n    _exit(1)\n"
        f"_write(_stdout, {marker!r} + b'\\n')\n"
    )

def execute_in_docker(code, tests, image):
    # Explicit opt-in. Neither generated code nor test code runs on the host.
    name = 'llm-eval-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as directory:
        marker = ('EVAL_OK_' + uuid.uuid4().hex).encode()
        path = Path(directory) / 'case.py'
        path.write_text(harness_source(code, tests, marker), encoding='utf-8')
        path.chmod(0o644)
        Path(directory).chmod(0o755)
        command = ['docker', 'run', '--name', name, '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534', '--pids-limit', '32', '--memory', '128m', '--cpus', '1', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m', '-v', f'{directory}:/case:ro', image, 'python', '-I', '/case/case.py']
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
            if result.returncode in (125, 126, 127):
                raise RuntimeError('Docker infrastructure failed; no execution score produced')
            return result.returncode == 0 and result.stdout == marker + b'\n'
        except subprocess.TimeoutExpired:
            return False
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)

def score(rows, predictions, docker_image=None):
    if not rows:
        raise ValueError('Test split must not be empty')
    expected = {row['id'] for row in rows}
    by_id = {row['id']: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError('Predictions must contain exactly one completion for every test id')
    if docker_image:
        subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        subprocess.run(['docker', 'image', 'inspect', docker_image], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    valid = exact = tested = passed = 0
    for row in rows:
        code = code_only(by_id[row['id']]['completion'])
        try:
            if code:
                compile(code, '<completion>', 'exec')
                valid += 1
        except (SyntaxError, ValueError):
            pass
        exact += code == code_only(row['output'])
        if docker_image and row.get('tests'):
            tested += 1
            passed += execute_in_docker(code, row['tests'], docker_image)
    return {'samples': len(rows), 'syntax_valid_rate': valid / len(rows), 'exact_match_rate': exact / len(rows), 'execution_tested': tested, 'pass_at_1': passed / tested if tested else None, 'note': 'Syntax/exact match are not functional correctness; pass@1 covers only curated tests supplied in the dataset.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--splits', type=Path, default=Path('splits'))
    parser.add_argument('--docker-image', help='Opt in to execution using an already pulled, trusted Python image; prefer an immutable digest')
    args = parser.parse_args()
    print(json.dumps(score(verify_splits(args.splits)['test'], read_jsonl(args.predictions), args.docker_image), indent=2))
