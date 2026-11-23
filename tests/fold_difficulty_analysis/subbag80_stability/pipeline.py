"""Verified full diagnostic followed by user-authorized aggregate-only publication."""
from pathlib import Path
import json,subprocess,sys
from publish_completed import digest,publish

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def main():
    proof=json.loads((BASE/'validation/verification.json').read_text());assert proof['passed']
    sources=json.loads((BASE/'validation/source_sha256.json').read_text())
    def check():
        for filename,expected in sources.items():
            if digest(filename)!=expected:raise ValueError(f'Source drift: {filename}')
    result={}
    try:
        check();subprocess.run([sys.executable,'-u',str(BASE/'run.py')],cwd=ROOT,check=True);check()
        result['status']='publishing';(BASE/'validation/publish_status.json').write_text(json.dumps(result))
        commit=publish();result.update(status='completed_pushed',commit=commit)
        print('Validated and pushed',commit,flush=True)
    except Exception as error:result.update(status='failed_or_publication_pending',error=f'{type(error).__name__}: {error}');raise
    finally:(BASE/'validation/publish_status.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
