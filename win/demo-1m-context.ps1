# ===========================================================================
# ds4 ROCm gfx1151 — >1,048,576-token CONTEXT capability demonstration.
#
# Goal: prove ds4 can FILL (prefill) more than 1,048,576 tokens of context on
# the AMD Strix Halo box (Radeon 8060S iGPU, gfx1151, 96 GB VRAM split + ~32 GB
# OS RAM) and still GENERATE COHERENT output -- not merely allocate the KV
# buffer (the latter was already shown in CAMPAIGN-REPORT-gfx1151.md Phase C).
#
# This script is meant to be run by the ORCHESTRATOR on the GPU, serially. It
# does NOT run anything itself unless invoked. Three tiers are provided:
#
#   Tier 0  (smoke,  ~minutes)  : ~16k filled ctx, dumps generated text.
#   Tier 1  (VRAM,   ~hours)    : largest single real prompt (~180k tok) fully
#                                 filled + coherent gen. Completable in one sit.
#   Tier 2  (>1M,    ~1-2 DAYS) : synthetic >1,048,576-token prompt, fully
#                                 filled + coherent gen. THE HEADLINE DEMO.
#
# HONEST BLOCKER: memory is NOT the limit (1M KV ~= 13-18 GiB, fits the 96 GB
# split per Phase C). PREFILL TIME is. This box prefills at ~80 tok/s @2k
# falling to ~22 tok/s @32k and lower beyond; integrated over 0..1.05M tokens
# the average is roughly 5-15 tok/s, so FILLING 1,048,576 tokens takes on the
# order of 20-60 WALL-CLOCK HOURS. Budget Tier 2 as a multi-day single run.
#
# Usage (from anywhere; paths are absolute):
#   pwsh -File win\demo-1m-context.ps1 -Tier 0        # quick coherence smoke
#   pwsh -File win\demo-1m-context.ps1 -Tier 1        # ~180k filled (hours)
#   pwsh -File win\demo-1m-context.ps1 -Tier 2        # >1M filled (DAYS)
#   pwsh -File win\demo-1m-context.ps1 -Tier 2 -Managed   # add GTT spill route
# ===========================================================================
param(
    [int]$Tier = 0,
    [switch]$Managed,            # set DS4_CUDA_MANAGED=1 (full-UMA / GTT spill)
    [int]$GenTokens = 24,        # small greedy gen at the frontier
    [int]$CtxMax = 1100000       # Tier 2 fill target (> 1,048,576)
)

$ErrorActionPreference = 'Continue'

# --- paths ----------------------------------------------------------------
$wt    = "C:\Development\ds4-worktrees\diskkv"
$bench = Join-Path $wt "ds4-bench.exe"
$model = "C:\Development\ds4-rocm\ds4\gguf\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
$res   = "C:\Development\ds4-rocm\results"
$out   = Join-Path $wt "demo-out"
$null  = New-Item -ItemType Directory -Force -Path $out -ErrorAction SilentlyContinue

$P_HUGE = Join-Path $res "bench-prompt-huge.txt"     # ~180k tok, 1.05 MB
$P_BIG  = Join-Path $out "bench-prompt-1m.txt"       # synthesized > 1M tok

# --- required runtime env (per project notes) -----------------------------
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:PATH
$env:DS4_CUDA_COPY_MODEL_CHUNKED = "1"
$env:DS4_LOCK_FILE = Join-Path $wt "ds4.lock"
$env:DS4_BENCH_DUMP_GEN = "1"        # gated text dump (added to ds4_bench.c) -> coherence check
if ($Managed) { $env:DS4_CUDA_MANAGED = "1" } else { Remove-Item Env:DS4_CUDA_MANAGED -ErrorAction SilentlyContinue }

if (-not (Test-Path $bench)) { Write-Error "ds4-bench.exe not found at $bench -- build first: ROCM_PATH=... win/build-rocm.sh"; exit 2 }
if (-not (Test-Path $model)) { Write-Error "model not found at $model"; exit 2 }

function Run-Bench {
    param([string]$Label,[string]$Prompt,[int]$CtxStart,[int]$CtxMaxL,[int]$CtxAlloc)
    $csv = Join-Path $out "$Label.csv"
    $log = Join-Path $out "$Label.gen.txt"     # stderr -> contains the generated text
    $args = @(
        '--chat-prompt-file', $Prompt,
        '-m', $model,
        '--ctx-start', "$CtxStart",
        '--ctx-max',   "$CtxMaxL",
        '--ctx-alloc', "$CtxAlloc",
        '--gen-tokens', "$GenTokens",
        '-t', '4',
        '--csv', $csv
    )
    Write-Host "=== [$Label] ctx-start=$CtxStart ctx-max=$CtxMaxL ctx-alloc=$CtxAlloc gen=$GenTokens managed=$($Managed.IsPresent)"
    Write-Host "    bench args: $($args -join ' ')"
    $t0 = Get-Date
    & $bench @args 2> $log
    $el = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host "    elapsed ${el}s  ->  csv=$csv  gen-text(stderr)=$log"
    if (Test-Path $csv)  { Write-Host "    --- CSV ---"; Get-Content $csv | ForEach-Object { Write-Host "    $_" } }
    if (Test-Path $log)  { Write-Host "    --- last generated text (coherence check) ---"
                           Get-Content $log | Select-String '^ds4-bench: gen@' | Select-Object -Last 3 | ForEach-Object { Write-Host "    $_" } }
}

switch ($Tier) {

  0 {
    # Coherence smoke: ~16k filled, single frontier, fast. Proves the dump works
    # and output is coherent. Completes in a few minutes after model load.
    Run-Bench -Label "tier0-smoke-16k" -Prompt $P_HUGE -CtxStart 16384 -CtxMaxL 16384 -CtxAlloc (16384 + 4096)
  }

  1 {
    # Max pure-VRAM filled context using the largest REAL prompt we have.
    # The huge prompt is ~180k tokens; we fill to its full length in ONE
    # prefill interval (ctx-start == ctx-max) to avoid intermediate gen cycles.
    # Expected: prefill many minutes-to-hours (22 tok/s falling), then a small
    # coherent gen. Pick a frontier <= the prompt's real token count.
    $frontier = 170000     # safely under the huge prompt's ~180k tokens
    Run-Bench -Label "tier1-vram-170k" -Prompt $P_HUGE -CtxStart $frontier -CtxMaxL $frontier -CtxAlloc ($frontier + 4096)
  }

  2 {
    # >1,048,576 FILLED tokens. Requires a prompt that TOKENIZES past CtxMax,
    # because ds4-bench slices a real token sequence and needs prompt.len>=ctx-max.
    # Synthesize one by repeating the huge prompt (~180k tok) ~8x -> ~1.4-1.9M tok.
    if (-not (Test-Path $P_BIG)) {
        Write-Host "Synthesizing >1M-token prompt at $P_BIG (concatenating huge prompt x8)..."
        $chunk = Get-Content -Raw $P_HUGE
        $sb = [System.Text.StringBuilder]::new()
        for ($i=0; $i -lt 8; $i++) { [void]$sb.Append($chunk); [void]$sb.Append("`n`n") }
        [System.IO.File]::WriteAllText($P_BIG, $sb.ToString())
        $mb = [math]::Round((Get-Item $P_BIG).Length/1MB,1)
        Write-Host "  wrote $P_BIG ($mb MB). Expected ~1.4-1.9M tokens (>1,048,576)."
    }
    Write-Warning "Tier 2 prefills $CtxMax tokens. At this box's prefill rate this is a MULTI-DAY single run (~20-60h). Ensure the orchestrator's timeout allows it (or it WILL be killed mid-prefill)."
    # Single giant fill interval: ctx-start == ctx-max so we prefill straight to
    # the >1M frontier once, then a tiny coherent gen. ctx-alloc has headroom.
    Run-Bench -Label "tier2-over1m" -Prompt $P_BIG -CtxStart $CtxMax -CtxMaxL $CtxMax -CtxAlloc ($CtxMax + 8192)
  }

  default { Write-Error "unknown -Tier $Tier (use 0, 1, or 2)"; exit 2 }
}
