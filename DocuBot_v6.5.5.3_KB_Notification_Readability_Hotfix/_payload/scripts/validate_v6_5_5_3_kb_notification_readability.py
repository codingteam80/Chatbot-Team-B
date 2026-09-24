from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs' / 'v6_5_5_3_notification_readability' / 'validation_latest.json'
checks = []


def add(name, passed, detail=''):
    checks.append({'name': name, 'passed': bool(passed), 'detail': str(detail)})


def read(rel):
    return (ROOT / rel).read_text(encoding='utf-8-sig')

# Syntax/critical files
for rel in ['app.py', 'config/settings.py', 'scripts/kb_update_runner.py', 'scripts/update_qdrant_incremental.py']:
    try:
        compile(read(rel), rel, 'exec')
        add(f'Python syntax: {rel}', True)
    except Exception as e:
        add(f'Python syntax: {rel}', False, e)

app = read('app.py')
css = read('ui/styles.css')
settings = read('config/settings.py')
manifest_path = ROOT / 'FINAL_COMPONENT_MANIFEST_v6.5.5.3.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}

# Exact UI intent
add('Success notification title preserved', 'Knowledge Base is successfully updated.' in app)
add('Success notification exposes Added/Updated/Deleted counts only', all(x in app for x in ['Added:', 'Updated:', 'Deleted:']))
add('Success notification still hides technical details', not any(x in app[app.find('def _kb_update_success_message'):app.find('# ======================================', app.find('def _kb_update_success_message'))] for x in ['Qdrant', 'BM25', 'semantic Rule', 'chunks embedded']))
add('Toast width increased for readability', 'width: min(500px' in css and 'max-width: min(500px' in css)
add('Toast has explicit readable minimum height', 'min-height: 86px !important' in css)
add('Toast releases fixed height', 'height: auto !important' in css)
add('Toast releases max-height', 'max-height: none !important' in css)
add('Toast releases clipping overflow', 'overflow: visible !important' in css)
add('Toast markdown text is not line-clamped', '-webkit-line-clamp: unset !important' in css)
add('Responsive toast rule exists', '@media (max-width: 640px)' in css)

# Preserve chat-disable behavior from v6.5.5.2
add('Chat input remains disabled during KB update', 'kb_chat_disabled = bool(' in app and 'disabled=kb_chat_disabled' in app)
add('Server-side chat submit guard preserved', 'kb_update_in_progress' in app and 'question' in app)
add('Disabled composer CSS preserved', ':has(textarea:disabled)' in css)

# Architecture invariants
for key, expected in [('CHUNK_SIZE', '900'), ('CHUNK_OVERLAP', '150'), ('MIN_RETRIEVAL_SCORE', '0.55'), ('VECTOR_TOP_K', '10'), ('BM25_TOP_K', '10'), ('FINAL_TOP_K', '3')]:
    m = re.search(rf'^\s*{re.escape(key)}\s*=\s*([^#\r\n]+)', settings, re.M)
    add(f'Invariant {key} remains {expected}', bool(m and expected in m.group(1)), m.group(1).strip() if m else 'missing')
add('Exactly two MultiQuery alternatives preserved', bool(re.search(r'^\s*MULTI_QUERY_VARIANT_COUNT\s*=\s*2\b', settings, re.M)))
add('Generation model remains qwen2.5:7b', 'qwen2.5:7b' in settings)
add('Embedding model remains qwen3-embedding:8b', 'qwen3-embedding:8b' in settings)
add('Reranker remains BAAI/bge-reranker-v2-m3', 'BAAI/bge-reranker-v2-m3' in settings)

# Manifest scope
add('v6.5.5.3 manifest exists', manifest_path.is_file(), manifest_path)
add('Manifest version is v6.5.5.3', manifest.get('version') == 'v6.5.5.3', manifest.get('version'))
add('Manifest records no toast clipping', manifest.get('ui', {}).get('kb_update_success_notification_overflow_clipped') is False)
add('Manifest preserves incremental lifecycle', manifest.get('kb_update', {}).get('incremental_core_changed') is False)
add('Manifest forbids install-time KB rebuild', manifest.get('install_time_production_kb_rebuild') is False)

failed = [c for c in checks if not c['passed']]
result = {
    'version': 'v6.5.5.3',
    'validation': 'PASS' if not failed else 'FAIL',
    'failed_checks': len(failed),
    'check_count': len(checks),
    'checks': checks,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f"v6.5.5.3 validation: {result['validation']} ({len(checks)-len(failed)}/{len(checks)})")
for c in failed:
    print('FAIL:', c['name'], c['detail'])
raise SystemExit(0 if not failed else 1)
