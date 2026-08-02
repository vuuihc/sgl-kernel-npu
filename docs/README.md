# SGL-KERNEL-NPU Documentation

We recommend new contributors start from writing documentation, which helps you quickly understand codebase.

## Optimization Roadmaps

- [单机单卡 Ascend 950 性能优化贡献路线](developer_guide/ascend950_single_card_optimization_roadmap.md)

## Docs Workflow

### Update Documentation

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```
---
