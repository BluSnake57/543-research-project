"""Readable job summaries and measured progress shared by background work and setup."""
from pathlib import Path
import time

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QProgressBar, QTreeWidgetItem

DONE = {'completed', 'failed', 'cancelled', 'interrupted'}
NAMES = {
    'download:scope': 'Download Scope SAE', 'scope-verify': 'Verify Qwen and Scope checksums',
    'setup': 'Set up models and evidence', 'download:qwen': 'Download chat model',
    'download:embedding': 'Download evidence-search model', 'fever': 'Download Wikipedia evidence',
    'fever-index': 'Build / resume Wikipedia search index',
    'posts-preview': 'Read post dataset', 'posts-import': 'Import and attach posts',
    'index': 'Index local evidence', 'search': 'Search local evidence',
    'map-posts': 'Suggest dataset field mapping', 'estimate-graph': 'Estimate missing graph details',
    'qwen-smoke': 'Check model generation and tools', 'training': 'Train virality model',
    'twitter-import': 'Import Twitter research dataset',
}


def byte_size(value):
    value = max(0, value)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes}m {seconds}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes}m'


def job_summary(job, titles=None, now=None):
    """Progress is for the current transfer/phase, never a fabricated overall percentage."""
    status = job['status']; terminal = status in DONE
    events = job.get('events', [])
    relevant = {'stage', 'download', 'file', 'indexing', 'vector_index', 'extraction', 'runtime', 'prompt_progress', 'scope_progress', 'scope_status'}
    latest = next((e for e in reversed(events) if e['event'] in relevant), {})
    name = NAMES.get(job['kind'], job['kind'].replace('-', ' ').capitalize())
    if job['kind'].startswith('chat:'):
        ident = job['kind'].split(':', 1)[1]
        name = 'Generate response · '+(titles or {}).get(ident, 'Conversation')
    status_label = {'running': 'In progress', 'queued': 'Waiting', 'completed': 'Complete',
                    'failed': 'Failed', 'cancelled': 'Cancelled', 'interrupted': 'Interrupted'}.get(status, status)
    if job.get('cancelling') and not terminal:
        status_label = 'Cancelling'
    detail = latest.get('stage', 'Working…')
    percent = None; progress_label = 'Working…'
    event = latest.get('event')
    if event == 'download':
        done = latest.get('downloaded', 0); total = latest.get('total')
        detail = ('Resuming ' if latest.get('resumed') else 'Downloading ')+latest.get('file', 'file')
        detail += '\n'+byte_size(done)+(f' of {byte_size(total)}' if total else ' · total size unknown')
        if latest.get('speed', 0) > 0:
            detail += ' · '+byte_size(latest['speed'])+'/s'
        if latest.get('eta') is not None:
            detail += ' · about '+duration(latest['eta'])+' left'
        if total:
            percent = min(100, int(100*done/total)); progress_label = f'{percent}% of file'
    elif event in ('vector_index', 'extraction'):
        done = latest.get('indexed', latest.get('extracted', 0)); total = latest.get('total')
        detail = ('Building search vectors' if event == 'vector_index' else 'Extracting Wikipedia evidence')
        detail += '\n'+(f'{done:,} of {total:,} passages' if event == 'vector_index'
                         else f'{byte_size(done)} of {byte_size(total or 0)}')
        if total:
            percent = min(100, int(100*done/total)); progress_label = f'{percent}% of phase'
    elif event == 'indexing':
        detail = ('Already indexed ' if latest.get('unchanged') else
                  'Indexed ' if latest.get('file_complete', True) else 'Indexing ')+Path(latest.get('path', 'evidence')).name
        detail += f'\n{latest.get("chunks", 0):,} passages processed in this file'
        if latest.get('total_files'):
            done, total = latest.get('completed_files', 0), latest['total_files']
            detail += f' · {done:,} of {total:,} files complete'
            percent = min(100, int(100*done/total)); progress_label = f'{percent}% of files'
    elif event == 'scope_progress':
        detail = f"Explanation · Scope · {latest.get('tokens', 0):,} tokens measured"
    elif event == 'scope_status':
        detail = 'Explanation · Scope · ' + latest.get('status', 'running')
    elif event == 'prompt_progress':
        done, total = latest.get('processed', 0), latest.get('total')
        detail = f'Processing prompt · {done:,} of {total or 0:,} tokens'
        if total:
            percent = min(100, int(100*done/total)); progress_label = f'{percent}% of prompt'
    elif event == 'runtime':
        detail = latest.get('warning', 'Model loaded · preparing generation')
    elif latest.get('file'):
        detail += '\n'+latest['file']
    file_event = next((e for e in reversed(events) if e['event'] == 'file'), None)
    if file_event and not terminal:
        detail += f' · File {file_event["index"]} of {file_event["count"]}'
    if status == 'queued':
        detail = 'Waiting for the current task to finish. No download has started.' if job['kind'].startswith(('download:', 'runtime-download:')) or job['kind'] == 'setup' else 'Waiting for the background worker.'
        if job['kind'].startswith('chat:'):
            detail = 'Waiting for the current model request to finish. Indexing runs separately.'
        elif job['kind'].startswith('runtime-activate:'):
            detail = 'Waiting for the current response before switching engines.'
        progress_label = 'Waiting'
        percent = 0
    elif detail == 'Queued':
        detail = 'Starting task…'
    if job.get('cancelling') and not terminal:
        detail = 'Stopping the task. Waiting for the current operation to return.'
        progress_label = 'Cancelling'
    if terminal:
        percent = 100 if status == 'completed' else 0
        progress_label = status_label
        detail = {'completed': 'Finished successfully.', 'cancelled': 'Task cancelled.',
                  'interrupted': 'The app closed before this task finished. Retry to continue.'}.get(status, job.get('error', 'Task failed.'))
        download_task = job['kind'].startswith(('download:', 'runtime-download:')) or job['kind'] in ('setup', 'fever')
        if status == 'cancelled' and download_task:
            detail = 'Stopped. Partial downloads are kept; use Retry / resume task to continue.'
        if status == 'cancelled' and job['kind'] == 'fever-index':
            detail = 'Indexing stopped. The download, extracted files, and committed passages are kept. Build / resume when needed.'
        result = job.get('result')
        if status == 'completed' and isinstance(result, dict):
            if 'indexed' in result:
                detail = f'Indexed {result["indexed"]} files; {len(result.get("errors", []))} file errors.'
                if result.get('errors'):
                    status_label = 'Complete with issues'
            elif 'path' in result:
                detail = 'Saved '+Path(result['path']).name
            elif 'model' in result:
                detail = 'Chat model is ready.' if result['model'] == 'qwen' else 'Evidence-search model is ready.'
            elif job['kind'] in ('fever', 'fever-index') and result.get('index', {}).get('mode') == 'keyword-only':
                detail = 'Wikipedia evidence is ready for keyword search.'
            elif result.get('index_deferred') or result.get('fever', {}).get('index_deferred'):
                detail = 'Download saved. FEVER indexing is deferred; build / resume its index from Evidence when needed.'
            elif job['kind'].startswith('runtime-'):
                detail = 'Engine selected.' if result.get('activated') else 'Engine installed.'
                if result.get('activation_job'):
                    detail += ' Activation queued after current model work.'
                if result.get('devices'):
                    detail += '\nDetected: '+'; '.join(result['devices'])
        if status == 'failed':
            detail = job.get('error', 'Task failed. Open details for diagnostics.')
    end = job.get('finished', events[-1].get('time', job['created']) if events else job['created']) if terminal else (now if now is not None else time.time())
    return {'name': name, 'status': status_label, 'detail': detail, 'percent': percent,
            'progress_label': progress_label, 'elapsed': duration(end-job['created'])}


def update_job_rows(table, jobs, titles=None):
    """Update existing rows so selection, scrolling, and progress widgets survive polling."""
    rows = {table.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole): table.topLevelItem(i)
            for i in range(table.topLevelItemCount())}
    for index, job in enumerate(jobs):
        ident = job['id']; summary = job_summary(job, titles)
        item = rows.pop(ident, None)
        if item is None:
            item = QTreeWidgetItem(); item.setData(0, Qt.ItemDataRole.UserRole, ident)
            item.setSizeHint(0, QSize(0, 66)); table.insertTopLevelItem(index, item)
            bar = QProgressBar(); bar.setMinimumWidth(130); table.setItemWidget(item, 3, bar)
        for column, field in ((0, 'name'), (1, 'status'), (2, 'detail'), (4, 'elapsed')):
            item.setText(column, summary[field]); item.setToolTip(column, summary[field])
        bar = table.itemWidget(item, 3)
        if summary['percent'] is None:
            bar.setRange(0, 0)
        else:
            bar.setRange(0, 100); bar.setValue(summary['percent'])
        bar.setFormat(summary['progress_label']); bar.setToolTip(summary['detail'])
    for item in rows.values():
        table.takeTopLevelItem(table.indexOfTopLevelItem(item))
