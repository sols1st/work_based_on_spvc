"""Resume a task-local checkpoint download using validated HTTP byte ranges."""
import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
TOTAL = 1217586395
target = ROOT / 'dinov2_vitl14_pretrain.pth'
partial = ROOT / 'dinov2_vitl14_pretrain.parallel'
prefix = target.stat().st_size if target.exists() else 0
if prefix == TOTAL:
    print('Checkpoint already has the expected length.', flush=True)
    raise SystemExit(0)
if not 0 <= prefix < TOTAL:
    raise RuntimeError(f'Unexpected existing file length: {prefix}')
with partial.open('wb') as dst:
    if prefix:
        with target.open('rb') as src:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    dst.truncate(TOTAL)
fd = os.open(partial, os.O_WRONLY)
lock = threading.Lock()
received = [0]
started = time.monotonic()


def fetch(index, start, end):
    done = 0
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                URL, headers={'Range': f'bytes={start + done}-{end}'}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f'bytes {start + done}-{end}/{TOTAL}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise RuntimeError(f'Unexpected range response: {response.status}, {response.headers}')
                while True:
                    block = response.read(512 * 1024)
                    if not block:
                        break
                    if start + done + len(block) > end + 1:
                        raise RuntimeError('Range response is longer than requested')
                    offset = 0
                    while offset < len(block):
                        offset += os.pwrite(fd, block[offset:], start + done + offset)
                    done += len(block)
                    with lock:
                        received[0] += len(block)
            if done != end - start + 1:
                raise RuntimeError('Truncated range response')
            print(f'part {index:02d} complete ({done} bytes)', flush=True)
            return
        except Exception as error:
            print(f'part {index:02d}, attempt {attempt + 1}: {error}', flush=True)
            if done == end - start + 1:
                return
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 10))


workers = 16
remaining = TOTAL - prefix
print(f'Reusing {prefix} bytes; downloading {remaining} bytes in {workers} parts.', flush=True)
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch, i, prefix + remaining * i // workers,
                        prefix + remaining * (i + 1) // workers - 1)
            for i in range(workers)
        ]
        while not all(f.done() for f in futures):
            time.sleep(10)
            elapsed = time.monotonic() - started
            print(f'{prefix + received[0]}/{TOTAL} bytes, '
                  f'{received[0] / max(elapsed, 1) / 1024**2:.2f} MiB/s', flush=True)
        for future in futures:
            future.result()
    os.fsync(fd)
finally:
    os.close(fd)
assert partial.stat().st_size == TOTAL
os.replace(partial, target)
digest = hashlib.sha256()
with target.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
target.with_suffix('.sha256').write_text(digest.hexdigest() + '\n')
print(f'Complete: {target}; SHA256 {digest.hexdigest()}', flush=True)
