#!/usr/bin/env python3
import sys, platform, os, traceback

def try_import(name):
    try:
        mod = __import__(name)
        return mod, None
    except Exception as e:
        return None, e

def pkg_version(dist_name):
    try:
        import importlib.metadata as im
        return im.version(dist_name)
    except Exception:
        return "unknown"

errors = []

print("=== Basic environment ===")
print(f"Python: {platform.python_version()}")
print(f"Platform: {platform.platform()}")

# --- Core: verl ---
verl, e = try_import("verl")
if e:
    errors.append(("verl import", e))
print(f"verl import: {'OK' if not e else 'FAIL'}  version={pkg_version('verl')}")

# --- Torch + CUDA sanity ---
torch, e = try_import("torch")
if e:
    errors.append(("torch import", e))
else:
    print(f"torch import: OK  version={torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        try:
            x = torch.randn(1, device="cuda")
            print("CUDA tensor create: OK")
        except Exception as ce:
            errors.append(("CUDA small tensor", ce))
    else:
        print("NOTE: CUDA not detected; verl can still import, but training/inference backends typically expect CUDA 12.x+.")  # FYI

# --- Optional deps (don’t fail the test if missing) ---
for opt in ["vllm", "sglang", "pyarrow", "tensordict", "transformers"]:
    mod, e = try_import(opt)
    print(f"{opt} import: {'OK' if not e else 'missing/skip'}",
          (f"version={getattr(mod, '__version__', pkg_version(opt))}" if not e else ""))

# --- Light touch on Single Controller API (no heavy downloads) ---
sc_ok = True
try:
    from verl.single_controller import Worker, ResourcePool  # core module
    print("verl.single_controller import: OK")
except Exception as se:
    sc_ok = False
    errors.append(("single_controller import", se))

# --- If Ray is installed, try a 1-worker CPU-only group ---
ray, e = try_import("ray")
if e:
    print("ray import: missing/skip (this is fine for a basic smoke test)")
else:
    print(f"ray import: OK  version={ray.__version__}")
    try:
        from verl.single_controller.ray import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs
        ray.init(ignore_reinit_error=True, num_cpus=1, include_dashboard=False, logging_level="ERROR")
        pool = RayResourcePool(process_on_nodes=[1], use_gpu=True, max_colocate_count=1)
        cls = RayClassWithInitArgs(Worker)
        group = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=cls, bin_pack=True)
        print(f"RayWorkerGroup world_size: {group.world_size} (expected 1) — OK")
        ray.shutdown()
    except Exception as re:
        errors.append(("RayWorkerGroup init", re))

print("\n=== Result ===")
if errors:
    print("FAIL ❌  One or more checks failed:")
    for where, err in errors:
        print(f"- {where}: {type(err).__name__}: {err}")
        tb = "".join(traceback.format_exception_only(type(err), err)).strip()
        print(f"  hint: {tb}")
    sys.exit(1)
else:
    print("PASS ✅  verl appears to be installed and minimally functional.")
    sys.exit(0)
