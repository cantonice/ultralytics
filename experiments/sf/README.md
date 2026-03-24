# SF 实验工作流

该目录定义了 SF 模块实验的可复现工作流。

## 目录结构

- `configs/`：实验配置 YAML 文件
- `results/`：由脚本生成的汇总结果表

## 1) Dry run（配置校验）

```bash
python tools/train_experiment.py --exp experiments/sf/configs/sfparallel_yolo26n.yaml --dry-run
```

默认情况下，`--dry-run` 不会加载权重，可避免在配置检查时误触发权重下载。

## 2) 训练 Baseline 与 SF 模型

```bash
python tools/train_experiment.py --exp experiments/sf/configs/baseline_yolo26n.yaml
python tools/train_experiment.py --exp experiments/sf/configs/sfparallel_yolo26n.yaml
```

如果配置中的权重文件不存在，训练会快速失败。
只有在你明确需要远程下载时，才使用 `--allow-auto-download`。

## 3) 汇总所有实验运行结果

```bash
python tools/summarize_runs.py --runs-root runs/sf --output experiments/sf/results/summary.csv
```

## 命名规范

每个配置建议使用如下运行名称格式：

`det_<dataset>_<model>_<variant>_e<epochs>`

示例：

- `det_glass_yolo26n_baseline_e200`
- `det_glass_yolo26n_sfparallel_e200`

## 可复现性检查清单

- 在所有可比实验中固定 `seed`。
- 保持 `deterministic: true` 以确保严格可复现。
- 除消融因素外，其余训练设置保持一致。
- 不要覆盖历史运行结果；每次试验使用唯一 `name`。
- 使用 summary CSV 作为结果对比的统一数据来源。
