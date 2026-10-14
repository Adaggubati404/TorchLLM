# TorchLLM: cleanup checklist

## Files to delete from the repo
- None. The existing `.gitignore` already covers `checkpoints/`, `*.pt`, generated `.bin` files, `metrics.jsonl`, `.DS_Store` and Python caches.

## Lines to add to `.gitignore`
```gitignore
.vscode/
.idea/
```

## Add
- [ ] `README.md` (replace with the new one)
- [ ] `LICENSE`: the old README said "MIT" but there was no LICENSE file
- [ ] The About description from `descriptions.md`
- [ ] Fill in the TODO Results and Sample generation blocks with real numbers from a run

## Optional
- `configs/smoke.yaml` sets `tensor_log: true` while its comment says "Off by default". Either change the comment or set it to `false`. The new README describes the current behavior.
