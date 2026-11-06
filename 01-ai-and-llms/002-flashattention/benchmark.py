"""Small reproducible CPU benchmark; CSV + environment metadata, no GPU needed."""

import argparse
import csv
import io
import json
import os
import platform
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter


def elapsed_ms(fn, repeats):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples.append((perf_counter() - start) * 1000)
    return median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[128, 512, 1024])
    parser.add_argument('--blocks', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    if min(args.sizes + args.blocks + [args.dim, args.repeats, args.threads]) <= 0:
        parser.error('Sizes, blocks, dimension, repeats and threads must be positive.')

    # Set these BEFORE importing NumPy, so its BLAS backend can read them on load.
    thread_settings = {name: str(args.threads) for name in (
        'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    os.environ.update(thread_settings)
    import numpy as np
    from attention import dense_attention, tiled_attention

    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)
    rows = []
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    environment = {
        'run_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'platform': platform.platform(), 'processor': platform.processor(),
        'blas_config': config.getvalue(), 'requested_thread_environment': thread_settings,
        'thread_note': 'Thread counts requested via environment before NumPy import; actual backend thread counts are not independently measured.',
        'parameters': {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()},
        'memory_metric': 'Calculated bytes in ONE score matrix/tile; not peak RAM or memory traffic.',
    }
    print('method        N    block   median_ms    max_abs_error    score_buffer_KiB')
    for n in args.sizes:
        q, k, v = [rng.standard_normal((n, args.dim)).astype(dtype)
                   for _ in range(3)]
        # Same input values, evaluated in float64 as the accuracy reference.
        reference = dense_attention(q.astype(np.float64), k.astype(np.float64),
                                    v.astype(np.float64))
        cases = [('dense', 0, lambda: dense_attention(q, k, v))]
        cases += [('tiled', block,
                   lambda block=block: tiled_attention(q, k, v, block, block))
                  for block in args.blocks]
        for method, block, fn in cases:
            actual = fn()
            np.testing.assert_allclose(actual, reference,
                                       rtol=1e-4 if dtype.itemsize == 4 else 1e-10,
                                       atol=1e-5 if dtype.itemsize == 4 else 1e-12)
            error = actual.astype(np.float64) - reference
            max_error = float(np.max(np.abs(error)))
            relative_error = float(np.linalg.norm(error) /
                                   max(np.linalg.norm(reference), np.finfo(float).tiny))
            milliseconds = elapsed_ms(fn, args.repeats)
            score_bytes = (n * n if method == 'dense' else min(n, block)**2) * dtype.itemsize
            rows.append(dict(method=method, n=n, d=args.dim, block=block,
                             dtype=args.dtype, median_ms=milliseconds,
                             max_abs_error=max_error, relative_frobenius_error=relative_error,
                             score_buffer_bytes=score_bytes))
            print(f'{method:8} {n:6} {block:8} {milliseconds:11.3f} '
                  f'{max_error:16.3e} {score_bytes / 1024:19.1f}')

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'measurements.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print(f'\nSaved CSV and environment metadata to {args.out.resolve()}')


if __name__ == '__main__':
    main()
