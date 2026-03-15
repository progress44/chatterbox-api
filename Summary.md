# Chatterbox TTS Olares Deployment Summary

## Scope

This document summarizes the main issues encountered while turning the Chatterbox TTS service into a working Olares deployment backed by a public GHCR image and NVIDIA GPU support.

## 1. Initial Service Packaging

- Wrapped `resemble-ai/chatterbox` with a FastAPI service exposing:
  - `/health`
  - `/v1/models`
  - `/tts`
  - `/v1/audio/speech`
  - `/v1/audio/speech/upload`
- Added Docker packaging and integrated the service into the existing voice stack.
- Created an Olares app package and Helm chart for deployment.

## 2. CUDA Support Was Present in Code but Not in Deployment

- The API supported `cpu`, `cuda`, and `auto`.
- The initial container and compose setup were still CPU-oriented.
- Added a GPU build path with CUDA-enabled PyTorch and NVIDIA runtime wiring.

## 3. GHCR Publishing and Registry Access

- Built and pushed the image to GHCR.
- Confirmed `latest` and explicit GPU tags existed in GHCR.
- Initially configured private-registry pulls through Olares image secrets.
- Later removed registry secret requirements after making the GHCR image public.

## 4. Olares Install and Upgrade Validation Issues

- Hit Olares package validation failures due to malformed manifest YAML.
- Fixed `OlaresManifest.yaml` indentation and packaging structure.
- Added Olares install prompts when needed, then later removed them after the image became public.
- Repeatedly bumped chart versions to keep package artifacts aligned with the image and manifest changes.

## 5. Image Pull and Stale Image Problems

- Olares repeatedly reused older `latest` images.
- Confirmed this by comparing the running pod `imageID` against the expected GHCR digest.
- Updated the chart to use `imagePullPolicy: Always`.
- Used direct digest pinning with `kubectl set image` as the most reliable way to force rollout to the intended build.

## 6. Olares Admission Policy and Security Context Issues

- Olares rejected early pods because the container or init container was running as root.
- Fixed the deployment to run as non-root.
- Later aligned the image user and pod security context to UID/GID `1000`.
- This matched Olares better than the earlier `10001` image user.

## 7. Persistent Storage Permission Issues

- `hostPath` directories were created as `root:root`.
- The app ran as a non-root user and could not write Hugging Face and Torch cache directories.
- Attempted root init-container ownership fixes, but Olares admission blocked them.
- Briefly switched caches to `emptyDir` to avoid permission problems.
- Reverted to `hostPath` after the requirement to keep persistent storage.
- Conclusion: `hostPath` ownership cannot be declared in the chart and must be solved by node-side ownership or a trusted root init pattern.

## 8. HAMI / VRAM Allocation

- HAMI logs showed `CUDA_DEVICE_MEMORY_LIMIT_0=0m`.
- Determined that Olares metadata alone was not enough for VRAM slicing.
- Added explicit GPU resources to the pod:
  - `nvidia.com/gpu: 1`
  - `nvidia.com/gpumem: 12288`
- Kept the Olares GPU injection annotation as well.

## 9. Blackwell / RTX 5090 CUDA Compatibility

- The first GPU image used `torch 2.6.0 + cu124`.
- The RTX 5090 Laptop GPU reported `sm_120`.
- PyTorch warned that the installed build only supported up to `sm_90`.
- Upgraded the build path to:
  - PyTorch `2.9.1`
  - CUDA `12.8`
- Verified the new `latest` and `gpu-cu128` tags in GHCR once the build finished.

## 10. Local Docker Build Disk Pressure

- Multiple CUDA image builds failed with `No space left on device`.
- Causes included:
  - heavy CUDA wheel downloads
  - duplicate or conflicting dependency trees pulling additional PyTorch stacks
- Reduced build bloat by:
  - moving to a leaner runtime dependency set
  - installing `chatterbox-tts`, `s3tokenizer`, and `conformer` with `--no-deps`
  - pruning Docker builder cache
- After pruning, the `cu128` build completed and published successfully.

## 11. Runtime Import Failures from Missing Dependencies

- Several import-time crashes appeared only after deployment:
  - `einops` missing
  - `onnx` missing
- These were side effects of over-trimming the runtime image.
- Restored missing dependencies and republished the image each time.
- Key lesson: the image needed a true runtime import validation step after every dependency reduction.

## 12. Numba / Librosa Startup Problems

- The app initially failed during `librosa` import because of numba caching behavior in the container.
- Fixed by:
  - disabling numba JIT caching at startup
  - pinning compatible `numba` and `llvmlite` versions
  - setting explicit cache-related environment variables

## 13. Hugging Face Token and Cache Wiring

- Added Olares system environment wiring so the service can use:
  - `OLARES_USER_HUGGINGFACE_TOKEN`
  - `OLARES_USER_HUGGINGFACE_SERVICE`
- Mapped those values into the container as:
  - `HF_TOKEN`
  - `HF_ENDPOINT`
  - `HUGGING_FACE_HUB_TOKEN`
- Also configured Hugging Face cache directories under `/data`.

## 14. API and Observability Improvements

- Added structured operational logging and `X-Request-Id`.
- Logged request start, completion, model loading, and synthesis failures.
- Updated docs to make error correlation easier through `kubectl logs`.
- Added API documentation into the Olares package README and manifest.

## 15. Current State

- The service image is published to GHCR with CUDA `12.8` and PyTorch `2.9.1`.
- The chart is configured to:
  - request one GPU
  - request `12288` MiB of HAMI GPU memory
  - run as UID/GID `1000`
  - always pull `latest`
- The latest packaged Olares chart artifact is `1.0.23`.

## Main Lessons

- `latest` is not reliable without `Always` pull policy or digest pinning.
- Olares admission policy strongly shapes what can be done with init containers and permissions.
- `hostPath` persistence and non-root execution are a difficult combination without node-side ownership preparation.
- New GPU generations require explicit verification of PyTorch CUDA architecture support.
- Runtime import validation is mandatory after slimming Python dependencies.
