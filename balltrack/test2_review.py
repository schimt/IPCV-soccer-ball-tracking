"""Resume model-blind clip-2 quality review using explicit, fresh audit events."""
import argparse
import json
from types import SimpleNamespace

from . import annotate
from .common import read_rows, save_json
from .two_clip import DATA, GT, VIDEO, check, required_quality_frames, read, validate_reference, verify


def pending_frames(rows, plan, events, requested=None):
    confirmations = {}
    for event in events:
        if event.get('time_utc', '') >= plan['created_utc'] and event.get('action') in {'human_review', 'set_status'}:
            confirmations[event['frame']] = event['after']
    required = required_quality_frames(rows, plan)
    if requested is not None:
        required &= set(requested)
    keys = ['status', 'source', 'reviewed', 'cx', 'cy', 'x', 'y', 'w', 'h']
    return sorted(i for i in required if i not in confirmations or any(
        (confirmations[i].get(k) if confirmations[i].get(k) != '' else None) != rows[i][k] for k in keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--priority', action='store_true')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    record = verify()
    rows = validate_reference(GT, record['clips'][1]['video'])
    plan = read(DATA/'quality_plan.json')
    # Same frozen sample/boundary rule; supplemental visual flags are recorded separately.
    required = sorted(required_quality_frames(rows, plan))
    if required != plan['frames']:
        plan['frames'] = required
        save_json(DATA/'quality_plan.json', plan)
        text = '# Full quality queue\n\n| Frame | Reason |\n| --- | --- |\n'
        text += '\n'.join(f'| {i} | Required quality review |' for i in required)+'\n'
        (DATA/'quality_review.md').write_text(text, encoding='utf-8')
    audit = GT.with_suffix('.audit.jsonl')
    events = [json.loads(line) for line in audit.read_text(encoding='utf-8').splitlines()] if audit.exists() else []
    priority = read(DATA/'quality_priority.json')['frames'] if args.priority else None
    pending = pending_frames(rows, plan, events, priority)
    target = DATA/('quality_priority_pending.md' if args.priority else 'quality_pending.md')
    text = '# Remaining quality confirmations\n\nInspect each frame, then R or correct box/O/U/F. Q saves and exits.\n\n| Frame | Reason |\n| --- | --- |\n'
    text += '\n'.join(f'| {i} | Fresh manual confirmation required |' for i in pending)+'\n'
    target.write_text(text, encoding='utf-8')
    print(f'{len(pending)} remaining frames in this queue. {len(required)} total required; no inference or scoring.')
    if not pending:
        if not args.priority:
            check()
            print('Quality review passed. Evaluation has NOT been started.')
        else:
            print('Priority queue complete. Run without --priority for the remaining fixed sample.')
        return
    if not args.list_only:
        annotate.main(SimpleNamespace(video=str(VIDEO),output=str(GT),stride=1,pending_only=False,
                                      frames_file=str(target),init_only=False))


if __name__ == '__main__':
    main()
