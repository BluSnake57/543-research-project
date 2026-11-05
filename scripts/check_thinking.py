"""Check Thinking Off/On using the app's real Scope worker streaming path.

Uses pinned Qwen and Scope files in application storage, without downloading files.
Exit 0: both verified; 1: failed; 2: On accepted but no thinking was observed.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/backend'))
from sentinel_x.assistant import validate_answer
from sentinel_x.service import Service


def check_mode(service, enabled, timeout):
    messages = [
        {'role': 'system', 'content': 'Return a JSON object with summary and claims. '
         'The summary must contain only the simplified fraction; claims must be an empty array.'},
        {'role': 'user', 'content': 'A bag has 3 red balls and 2 blue balls. Two balls are drawn '
         'without replacement. What is the probability they have the same color?'}]
    def work(job):
        job.thinking_enabled = enabled
        return service.runtime.stream(messages, [], job)
    ident = service.jobs.submit('thinking-check', work)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        saved = service.jobs.get(ident)
        if saved['status'] in ('completed', 'failed', 'cancelled'):
            break
        time.sleep(.1)
    else:
        service.cancel(ident)
        return {'mode': 'On' if enabled else 'Off', 'passed': False,
                'detail': f'No completed response within {timeout} seconds.'}
    if saved['status'] != 'completed':
        return {'mode': 'On' if enabled else 'Off', 'passed': False,
                'detail': saved.get('error', saved['status'])}
    answer = validate_answer(saved['result']['content'], [])
    correct = answer['summary'].strip() == '2/5'
    thinking_seen = saved.get('thinking_characters', 0) > 0
    return {'mode': 'On' if enabled else 'Off',
            'passed': correct and thinking_seen == enabled,
            'inconclusive': enabled and correct and not thinking_seen,
            'answer_correct': correct, 'thinking_observed': thinking_seen,
            'thinking_seconds': saved.get('thinking_seconds', 0)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=300, help='Seconds allowed per mode, including model loading')
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error('Timeout must be positive')
    results = []
    service = Service(args.data_dir)
    if True:
        try:
            for enabled in (False, True):
                try:
                    results.append(check_mode(service, enabled, args.timeout))
                except Exception as exc:
                    results.append({'mode': 'On' if enabled else 'Off', 'passed': False, 'detail': str(exc)})
        finally:
            service.close(); service.jobs.shutdown(wait=True)
    print(json.dumps(results, indent=2))
    if all(result['passed'] for result in results):
        return 0
    if all(result['passed'] or result.get('inconclusive') for result in results):
        return 2
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
