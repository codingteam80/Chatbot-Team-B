from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs' / 'v6_5_5_4_notification_card' / 'validation_latest.json'
checks = []


def add(name, passed, detail=''):
    checks.append({'name': name, 'passed': bool(passed), 'detail': str(detail)})


def read(rel):
    return (ROOT / rel).read_text(encoding='utf-8-sig')


for rel in ['app.py', 'config/settings.py', 'scripts/kb_update_runner.py', 'scripts/update_qdrant_incremental.py']:
    try:
        compile(read(rel), rel, 'exec')
        add(f'Python syntax: {rel}', True)
    except Exception as e:
        add(f'Python syntax: {rel}', False, e)

app = read('app.py')
css = read('ui/styles.css')
settings = read('config/settings.py')
manifest_path = ROOT / 'FINAL_COMPONENT_MANIFEST_v6.5.5.4.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}

success_fn = app[app.find('def _kb_update_success_message'):app.find('# ======================================', app.find('def _kb_update_success_message'))]
render_fn = app[app.find('def _render_kb_update_flash'):app.find('# ======================================', app.find('def _render_kb_update_flash'))]

add('Success marker uses structured counts', 'KB_UPDATE_SUCCESS::' in success_fn)
add('Success card title is exact', 'Knowledge Base is successfully updated.' in render_fn)
add('Success card renders Added count', 'Added:' in render_fn)
add('Success card renders Updated count', 'Updated:' in render_fn)
add('Success card renders Deleted count', 'Deleted:' in render_fn)
add('Success card hides technical details', not any(x in render_fn for x in ['Qdrant', 'BM25', 'semantic Rule', 'chunks embedded', 'unchanged skipped']))
add('Success no longer uses native Streamlit toast', 'st.toast(flash_message' not in render_fn.split('return')[0])
add('Failure warning toast remains available', 'st.toast(flash_message, icon="⚠️")' in render_fn)
add('Success card is fixed overlay', '.docubot-kb-update-notification {' in css and 'position: fixed !important' in css)
add('Success card is top-right', 'right: 24px !important' in css and 'top: 92px !important' in css)
add('Success card has readable width', 'width: min(470px' in css)
add('Success card body overflow is visible', 'overflow: visible !important' in css)
add('Success counts wrap safely', '.docubot-kb-update-notification__counts {' in css and 'flex-wrap: wrap !important' in css)
add('Success counts use distinct count chips', '.docubot-kb-update-notification__counts span {' in css)
add('Success card is responsive', '@media (max-width: 640px)' in css and 'width: calc(100vw - 24px) !important' in css)
add('Success card does not modify page flow', 'position: fixed !important' in css)

add('Chat input remains disabled during KB update', 'kb_chat_disabled = bool(' in app and 'disabled=kb_chat_disabled' in app)
add('Server-side chat submit guard preserved', 'kb_update_in_progress' in app and 'question' in app)
add('Disabled composer CSS preserved', ':has(textarea:disabled)' in css)
add('Maintenance panel stability preserved', '.st-key-kb_maintenance_panel' in css)

for key, expected in [('CHUNK_SIZE', '900'), ('CHUNK_OVERLAP', '150'), ('MIN_RETRIEVAL_SCORE', '0.55'), ('VECTOR_TOP_K', '10'), ('BM25_TOP_K', '10'), ('FINAL_TOP_K', '3')]:
    m = re.search(rf'^\s*{re.escape(key)}\s*=\s*([^#\r\n]+)', settings, re.M)
    add(f'Invariant {key} remains {expected}', bool(m and expected in m.group(1)), m.group(1).strip() if m else 'missing')
add('Exactly two MultiQuery alternatives preserved', bool(re.search(r'^\s*MULTI_QUERY_VARIANT_COUNT\s*=\s*2\b', settings, re.M)))
add('Generation model remains qwen2.5:7b', 'qwen2.5:7b' in settings)
add('Embedding model remains qwen3-embedding:8b', 'qwen3-embedding:8b' in settings)
add('Reranker remains BAAI/bge-reranker-v2-m3', 'BAAI/bge-reranker-v2-m3' in settings)

add('v6.5.5.4 manifest exists', manifest_path.is_file(), manifest_path)
add('Manifest version is v6.5.5.4', manifest.get('version') == 'v6.5.5.4', manifest.get('version'))
add('Manifest records custom success card', manifest.get('ui', {}).get('kb_update_success_notification') == 'docubot_fixed_notification_card')
add('Manifest records native success toast disabled', manifest.get('ui', {}).get('kb_update_success_notification_native_streamlit_toast_for_success') is False)
add('Manifest preserves incremental lifecycle', manifest.get('kb_update', {}).get('incremental_core_changed') is False)
add('Manifest forbids install-time KB rebuild', manifest.get('install_time_production_kb_rebuild') is False)

failed = [c for c in checks if not c['passed']]
result = {
    'version': 'v6.5.5.4',
    'validation': 'PASS' if not failed else 'FAIL',
    'failed_checks': len(failed),
    'check_count': len(checks),
    'checks': checks,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f"v6.5.5.4 validation: {result['validation']} ({len(checks)-len(failed)}/{len(checks)})")
for c in failed:
    print('FAIL:', c['name'], c['detail'])
raise SystemExit(0 if not failed else 1)
