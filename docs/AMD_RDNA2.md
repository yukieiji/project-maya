# RX 6800 / 6900 and Radeon PRO W6800: experimental Maya HIP on RDNA2

Maya's HIP backend also builds and runs for Radeon RX 6800 / 6900 and the Radeon PRO W6800 (`gfx1030`, Navi 21): one
GPU, text input, Linux and Windows. RDNA2 is the same wave32 family with the same 64 KiB of LDS a workgroup as the RDNA3
cards Maya is measured on - hipInfo says so on the W6800 (`sharedMemPerBlock: 64.00 KB`) - so the kernels that need
those two things are the kernels gfx1100 runs. What differs is one instruction and a few dispatch choices, listed under
[What an RDNA2 card runs](#what-an-rdna2-card-runs).

```sh
./maya.sh --backend hip --gpu 0 --check          # Linux
START-MAYA.bat --backend hip --gpu 0 --check     # Windows
```

Setup takes the card through the kernel's KFD topology (Linux) or ROCm's hipInfo (Windows); before the ROCm SDK is in,
Windows reads the display adapter from the registry - by its PCI device ID where Maya has one (the Radeon PRO W6800,
`73a3`) and else by its name, which knows the desktop RX 6800 / 6900 / 6950 XT. An adapter neither names reads as
"unknown" until hipInfo takes the architecture off the card itself. An RX 6700 / 6600 is `gfx1032` / `gfx1034`,
another RDNA2 stepping: it builds with CMake but the installer does not offer it, and an iGPU (`gfx1036`) stays
refused.

## ROCm 7, next to an AMD HIP SDK 6.x

Maya compiles with **ROCm 7 or newer** and nothing else; the port has never been built with a 6.x compiler. On Windows
that does not mean uninstalling what is already there:

- setup looks through `ROCM_PATH`, TheRock's `rocm-sdk`, `HIP_PATH` and `C:\Program Files\AMD\ROCm\*` and **takes a 7
  or newer when one is installed**;
- when only a 6.x SDK is here, it says so and offers AMD's ROCm SDK wheels (`rocm[libraries,devel,device-gfx1030]`,
  pinned to `THEROCK_VERSION`) into Maya's `.venv`. The SDK installed on the system keeps working for everything else,
  and the engine then runs on the `.venv`'s `amdhip64_7.dll` next to `strata.exe` rather than the driver's copy in
  `System32`;
- `ROCM_PATH` wins over all of that, so it must not point at the 6.x SDK. `ROCM_VENV` names another venv with the
  wheels.

On Linux, a system ROCm 7 with its HIP compiler and hipBLAS (`ROCM_PATH` for another root), as in
[AMD_MAYA.md](AMD_MAYA.md).

## One binary for the card in this PC

Setup compiles the engine for the architectures of the GPUs it picked, not for every architecture Maya knows: an RX
6800 gets `gfx1030` alone, and a split of an RX 7900 XT with an RX 6800 gets `gfx1030;gfx1100`. `--setup` after a card
is swapped compiles it again for the card it now sees. A manual build with Maya's pinned llama.cpp checkout:

```sh
maya_rocm_root="$(realpath "${ROCM_PATH:-/opt/rocm}")"
cmake -S . -B build-rdna2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_BUILD_TESTS=ON \
  '-DCMAKE_HIP_ARCHITECTURES=gfx1030' \
  -DCMAKE_HIP_COMPILER="$maya_rocm_root/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$maya_rocm_root" \
  -DCMAKE_PREFIX_PATH="$maya_rocm_root" \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-rdna2 -j 3
```

On Windows the same configure line as `build-maya-hip.bat` (which setup writes), or
`tools\hip\build_maya_windows.bat` with `STRATA_HIP_ARCHS=gfx1030`.

## What an RDNA2 card runs

| Piece | RDNA2 (gfx1030) |
| --- | --- |
| quantized dot product | `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`), the signed form RDNA2 has; RDNA3's `sudot4` does not exist there. Same signed x signed, modulo 2^32 contract as CUDA's `__dp4a`, checked against a CPU reference by `hip_intrinsics` |
| LDS a workgroup | 64 KiB, as on RDNA3: the kernels' shared-memory budgets are unchanged |
| wave size | wave32, which the engine requires; a runtime reporting wave64 is refused at startup with that said in words |
| prompt attention | the F32 kernel `mla_attn`. ROCm 7's rocWMMA has no RDNA2 in its own architecture list (gfx9xx, gfx11xx, gfx12xx - its `config.hpp` stops with "Unsupported architecture" on `gfx1030`), so the FP16 `mla_attn_f16q` is not built for this card, and neither is `mla_attn_wmma2`, whose intrinsics are gfx11/gfx12 only. `STRATA_GLM_PREFILL_ATTN` asks for another path but gets this one, and `STRATA_GLM_PREFILL_ATTN_CHECK=1` compares a tensor-core path with it where there is one |
| decode experts | the original HIP expert kernels; the packed-sign ones measured on gfx1100 are dispatched for gfx1100/gfx1151 only ([AMD_EXPERT_KERNELS.md](AMD_EXPERT_KERNELS.md)) |
| dense projections | plain hipBLAS: the tuned hipBLASLt solution tables here are for gfx1100/gfx1151/gfx1201, and the engine takes a table only for its own architecture and hipBLASLt version |
| prompt MMQ | llama.cpp's ggml kernels, whose AMD compute-capability the architecture name maps to RDNA2 |

## Validation

The RDNA2 code paths are the ones the HIP checks cover, and they run on the card. `HIP_VISIBLE_DEVICES` names the card
by its HIP number - on a PC with an iGPU next to it the checks otherwise run on the iGPU and stop with "is not an
architecture this Maya engine was compiled for":

```sh
python3 tools/test_maya_hip.py
python3 tools/test_hip_archs.py
env -u ROCR_VISIBLE_DEVICES LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  ctest --test-dir build-rdna2 --output-on-failure --timeout 120 \
  -R '^(glm_(hc|kda|ffn|dsa|layer)_parity|glm_model_test|hip_intrinsics|hip_(glm_handoff|glm_prefill_attention|handoff|device_selftest|prefill_mmq_parity|expert_cache_staging)|iq1_s_parity)$'
```

On Windows the same list, in a shell with Visual Studio's environment and the SDK's `bin` on PATH (what
`build-maya-hip.bat` sets up), with `-DSTRATA_BUILD_TESTS=ON` on the configure line and `HIP_VISIBLE_DEVICES` set to
the RDNA2 card's hipInfo number:

```bat
set HIP_VISIBLE_DEVICES=1
.venv\Scripts\ctest.exe --test-dir build-hip --output-on-failure --timeout 120 -R "^(glm_(hc|kda|ffn|dsa|layer)_parity|glm_model_test|hip_intrinsics|hip_(glm_handoff|glm_prefill_attention|handoff|device_selftest|prefill_mmq_parity|expert_cache_staging)|iq1_s_parity)$"
```

`hip_intrinsics` is the RDNA2-specific one: it compares `__dp4a` (here `v_dot4_i32_i8`), the packed-byte subtract and
compare, the shuffles and the ballot against CPU references, and it stops with "expected wave32" when the runtime
hands the engine wave64. `hip_glm_prefill_attention` compares the prompt attention against the FP32 kernel, which is
the check to look at if a prompt answer goes wrong on an RDNA2 card; without rocWMMA it prints its FP16 half as
skipped.

**Measured:** all fourteen of those checks passed in October 2026 on a Radeon PRO W6800 (30 GB, `gfx1030`) with AMD's
ROCm 7.14.1 wheels on Windows - `hip_intrinsics` (the dot4 and the wave32) among them. What is still missing is the
model level: decode and prompt tokens/s, and the answer-quality spot checks, are the ones from the report in issue
#311 and from whoever runs a model first. Send `./maya.sh --backend hip --bench` and `--bench`'s follow-up `--report`
(Windows: `START-MAYA.bat --backend hip --bench` / `--report`) and this page gets the table. Until then treat the
RDNA2 numbers as unknown rather than as the gfx1100 ones: an RX 6800 XT has 16 GB where the 7900 XT has 20, and the
expert pool follows the VRAM.

## When it does not start

- **"is not an architecture this Maya engine was compiled for"**: the engine in `build-hip/` was compiled for another
  card. `./maya.sh --backend hip --setup` compiles it for this one; the message names the architectures in the binary.
- **"runs wave64; Maya's HIP kernels need wave32"**: the HIP runtime handed this card wave64. Say so in the report -
  the engine then needs its wave32 target feature, which is a build-argument change, not a kernel change.
- **prompts wrong or slow**: this card already runs the FP32 attention (above), so `STRATA_GLM_PREFILL_ATTN` has
  nothing to switch; `STRATA_GLM_MLA_F16=0` also takes the FP16 products out, and
  `STRATA_GLM_PREFILL_ATTN_CHECK=1` compares a tensor-core path with the FP32 one where the binary has both.
- **experts**: `STRATA_HIP_EXPERTS_LEGACY=1` is a no-op here - RDNA2 already runs the original kernels.
- **"`__device__ function 'isless' cannot overload __host__ __device__ function`"**: a Visual Studio newer than 2022
  17.12 and ROCm's math wrappers disagree (its `<cmath>` declares `isless()` and friends `constexpr`, which clang's HIP
  mode reads as host+device, and those cannot be overloaded). Setup compiles a two-line `.hip` file, sees that, and
  builds with `-Xclang -fno-cuda-host-device-constexpr`; by hand, add that to `CMAKE_HIP_FLAGS`. [AMD_MAYA.md](AMD_MAYA.md)
  says where.
