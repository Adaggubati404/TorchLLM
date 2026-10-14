# TorchLLM: cleanup checklist (updated after your Oct 14 commit)

Already done: `LICENSE` added. No stray files are tracked.

## Still to do
- [ ] Delete `CLEANUP.md` from the repo. It's a note for you, not part of the project.
- [ ] Replace `README.md` with the new version.
- [ ] Optional: `configs/smoke.yaml` sets `tensor_log: true` while its comment says "Off by default". Fix the comment or the value.

```bash
git rm CLEANUP.md
```

## Lines to add to `.gitignore`
```gitignore
.vscode/
.idea/
```
