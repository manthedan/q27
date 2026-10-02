# Host DMA corruption diagnostics (2026-08-23/24)

Instruments built to localize silent single-bit corruption of model loads on
`haight`. Kept because the fault is NOT fixed -- these are what to re-run after
any hardware change (reseat, BIOS, DIMM population).

**State of the diagnosis:** real corruption, ~1 event per 4.5 TB transferred,
bursty and non-stationary. Excluded by measurement: the card (both GPUs flip
at equal rates after a physical slot swap), the slot, VRAM (both device
regions flip), the checksum itself (recomputes agree), memory clock (7001 vs
14001 MHz, p=0.55), host page-cache placement, host DRAM holding data, the
PCIe link (Replays Since Reset = 0), and cache residency (a FLUSHED arm flips
too). It is bidirectional -- D2H readbacks corrupt as readily as H2D uploads.

**The surviving invariant, over 23 localized events on two GPUs, six
instruments, three engines:** every flip is at **dword bit 21 or bit 27**,
never any other position. That is a lane-level signature on a 32-bit-wide
path, which is what points at the host memory/fabric side rather than at
either GPU. See `docs/BUILDLOG.md` and the `hardware_5090_silent_corruption`
memory entry.

## Build

    nvcc -O2 -arch=sm_120 -o locate locate.cu            # generic H2D + readback diff
    nvcc -O2 -arch=sm_120 -o pin    pin.cu               # pinned vs pageable, both directions
    nvcc -O2 -arch=sm_120 -Xcompiler -mavx2 -o hot hot.cu    # cache-hot vs non-temporal staging
    nvcc -O2 -arch=sm_120 -Xcompiler -mavx2 -Xcompiler -mclflushopt -o hot3 hot3.cu  # per-CCD + flushed
    nvcc -O2 -arch=sm_120 -o addr  addr.cu               # same data -> two device regions
    g++  -O2 -std=c++17 -I llama.cpp/include -I llama.cpp/ggml/include \
         llama_load_readback.cpp -o llcheck -lllama -lggml -lggml-base

(The host-DRAM-only scan and the `hot4` arm runner referenced below were
never committed; only the six sources above are in this directory.)

`Q27_WSUM_LOCATE=1` (with `Q27_PRINT_WSUM=1`) does the same for q27's own
loader, per tensor.

## Two rules learned the hard way

1. **CUDA orders devices fastest-first; nvidia-smi orders by bus.** After the
   slot swap, `CUDA_VISIBLE_DEVICES=0` is the 5090 while `nvidia-smi` index 0
   is the 3090. A script that assumes they match runs on the wrong card and
   mislabels its arms.
2. **Stop on evidence, not on a round count.** The rate is bursty; three
   fixed-length runs ended in quiet windows and produced non-results. Run the
   known-flipping arm as a same-window control and stop when IT has fired N
   times (`hot4` pattern). A clean arm means nothing unless the control fired.
