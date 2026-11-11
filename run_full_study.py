"""Run the frozen extended study serially, retaining a durable progress log."""
from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'study_v2'
OUT.mkdir(parents=True, exist_ok=True)

with (OUT / 'pipeline.log').open('a', encoding='utf-8', buffering=1) as log:
    for stage in ('tune', 'main', 'external', 'width', 'evaluate'):
        status = {'stage': stage, 'started_utc': datetime.now(timezone.utc).isoformat(), 'state': 'running'}
        (OUT / 'pipeline_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        line = 'STAGE ' + stage
        print(line, flush=True)
        log.write(line+'\n')
        process = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', 'study_runner.py', stage],
                                   cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8')
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
        status.update(state='complete' if code == 0 else 'failed', exit_code=code,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        (OUT / 'pipeline_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        if code:
            raise SystemExit(code)
    (OUT / 'pipeline_status.json').write_text(json.dumps({'state':'complete','stage':'all',
        'finished_utc':datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
print('ALL EXTENDED STUDY STAGES COMPLETE', flush=True)
