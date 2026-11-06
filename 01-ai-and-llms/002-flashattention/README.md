# Tiled Attention and Online Softmax on a CPU

An educational forward-pass implementation of scaled dot-product attention.
It compares an explicit dense baseline with a tiled algorithm inspired by
FlashAttention. It runs on the CPU using NumPy and examines how block size
affects the size of intermediate score arrays, numerical agreement and runtime.

## Scope

- Q, K, V have shape `(N, d)` and the same float32 or float64 dtype.
- One attention head, batch size one, no mask, dropout or backward pass.
- Inputs must be finite and must not overflow during dot products.
- The same mathematical attention function is evaluated, with floating-point
  differences allowed. The tiled version does not materialize full score or
  probability matrices when the blocks are smaller than N.
- This Python implementation does not reproduce GPU kernel fusion, shared-memory
  management or GPU performance. It still performs quadratic attention work.

## Run

From the repository root, create an environment once if needed:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\01-ai-and-llms\002-flashattention\requirements.txt
```

Change to this topic folder and run correctness checks and the small benchmark:

```powershell
cd .\01-ai-and-llms\002-flashattention
..\..\.venv\Scripts\python.exe -m unittest -v
..\..\.venv\Scripts\python.exe benchmark.py
```

Reproduce the block-size sweep discussed below:

```powershell
..\..\.venv\Scripts\python.exe benchmark.py --blocks 16 32 64 128 --out results/block-sweep
```

An optional larger run (writes to the default results directory):

```powershell
..\..\.venv\Scripts\python.exe benchmark.py --sizes 128 512 1024 2048 --blocks 32 64 128 --repeats 7
```

Use `--out results/another-run` to preserve a separate run. Use `--dtype float64`
to compare a higher-precision configuration. Keep `--threads` fixed across runs.

## Read the Implementation

Scaled dot-product attention computes $S = QK^\top / \sqrt{d}$,
$P = \operatorname{softmax}(S)$ row by row, then $O = PV$.
The dense baseline stores the full score matrix $S$ and weight matrix $P$.
The tiled implementation computes a piece of $S$, updates the output
accumulator, and discards the tile before moving on.

Start with `safe_softmax` and `dense_attention` in `attention.py`. Then follow
`tiled_attention`: the outer loop selects query rows; the inner loop scans all
key/value blocks. Each query row keeps:

- `m`: maximum score seen so far.
- `l`: sum of exponentials relative to that maximum.
- `acc`: value-weighted numerator in the same scale.

For each new tile, `alpha = exp(old_m - new_m)` rescales BOTH old accumulators.
The score tile is overwritten with `exp(scores - new_m)`, which is not yet a
normalized probability matrix. Divide `acc / l` only after all keys are processed.

For example, take one query row with two scores, $0$ and $\ln 2$, and scalar
values $2$ and $8$. Process each score as its own block. After the first block,
`m = 0`, `l = 1` and `acc = 2`. The second score raises the maximum, so
$\alpha = e^{0 - \ln 2} = 1/2$. The second score's exponential is $1$ in the
new scale. The updated denominator is $1/2 + 1 = 1.5$ and the updated
numerator is $2/2 + 8 = 9$. The output is $9/1.5 = 6$, the same as
$(1/3)\cdot 2 + (2/3)\cdot 8$ from dense softmax.

Stable softmax avoids overflow in the exponential by subtracting the row maximum.
Negligibly small terms can still underflow to zero. Completely masked rows and
infinite input logits are outside the scope of this example.

## Tests and Measurements

`test_attention.py` checks random seeds and dtypes, incomplete blocks, zero queries,
constant values, softmax shift invariance, and a larger maximum arriving in a later
block. Zero-query and constant-value tests have answers independent of the dense
implementation. The benchmark also checks every output against a float64 dense
reference evaluated on exactly the same input values.

`benchmark.py` uses three warmups and the median of repeated timings. Input
generation, correctness checking and file writing are outside the timed call;
allocations inside each implementation are included. BLAS thread counts are
requested through environment variables before NumPy is imported; actual backend
thread counts are not independently measured. Each run saves its measurements
and environment to the output directory.

**Memory metric:** `score_buffer_bytes` is a calculated size of ONE dense score
matrix or ONE tile. It is not a measured peak and excludes probability matrices,
other temporaries, Q/K/V, output, BLAS workspace and allocator overhead. It does not
measure data movement or establish an equal reduction in total RAM consumption.
`block=0` denotes the dense baseline, which has no block-size parameter.

## Block-Size Sweep

The sweep used FP32 inputs, head dimension $d=64$, sequence lengths
$N\in\{128,512,1024\}$ and seven timed repetitions per configuration. The table
holds $N=1024$ fixed so the methods are compared on the same input.
See the [full CSV](results/block-sweep/measurements.csv) and
[run configuration](results/block-sweep/environment.json) for all 15 cases.

| Method | Median time (ms) | One score array (KiB, calculated) | Max absolute error vs. FP64 |
| --- | ---: | ---: | ---: |
| Dense | 9.71 | 4096 | 1.68e-7 |
| Tiled, 16 × 16 | 75.12 | 1 | 1.23e-7 |
| Tiled, 32 × 32 | 28.47 | 4 | 1.08e-7 |
| Tiled, 64 × 64 | 11.98 | 16 | 1.23e-7 |
| Tiled, 128 × 128 | 9.18 | 64 | 1.40e-7 |

All configurations passed the numerical tolerance check. Across the sweep,
the largest absolute error was below $3.2\times10^{-7}$.

At $N=1024$, a 128 × 128 score tile is 64 times smaller than the full score
matrix, with similar measured CPU runtimes in this run. Smaller tiles use even
less space per score array but ran more slowly. One possible explanation is
that 16 × 16 tiles require 4,096 tile updates, compared with 64 updates for
128 × 128 tiles. Each update incurs Python loop and NumPy call overhead, while
both configurations still compute scores for the same token pairs. The
benchmark does not isolate the cost of those updates.

At $N=128$ with a 128 × 128 tile, both methods produce a full-size score array.
Any timing difference in that case cannot be attributed to smaller score tiles.
The implementations differ in temporary allocations and where they normalize:
the dense version normalizes the weights before multiplying by $V$, while the
tiled version normalizes the output accumulator at the end.

FlashAttention uses tiling and online normalization to combine score
computation, softmax and output accumulation in a GPU kernel. This lets it
avoid writing full score and probability matrices to GPU high-bandwidth
memory between steps. This experiment demonstrates the same mathematical
idea on a CPU. It does not measure GPU runtime, memory traffic, or total peak
memory, and the small timing difference at the largest tile does not establish
a reproducible speedup.

## References

1. Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), 2022, Section 3.1.
2. Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023, Section 3.1.1.
3. Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018, Section 3.

## Related Topics

- [Transformer Architecture](../001-transformer-architecture/)
- [Matrix Multiplication Tiling](../../10-gpu-and-hpc/094-matrix-multiplication-tiling/)
- [Kernel Fusion](../../10-gpu-and-hpc/100-kernel-fusion/)
- [Microbenchmarking](../../06-performance-engineering/052-microbenchmarking/)
