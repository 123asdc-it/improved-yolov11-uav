"""
collect_results.py — 一键收集所有实验结果，生成论文表格数据

运行时机：所有实验完成后
用法：python scripts/collect_results.py

输出：
  - outputs/all_results.json        完整结果
  - outputs/table_ablation.txt      消融表（直接复制到论文）
  - outputs/table_sota.txt          SOTA对比表
  - outputs/latex_tables.tex        所有表格的 LaTeX 代码
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 服务器上的实际路径（路径嵌套问题）
# runs/detect/runs/detect/ 是已知的路径嵌套
SERVER_RUNS_BASE = Path("runs/detect/runs/detect")
LOCAL_RUNS_BASE = Path("runs")


def find_best_pt(exp_name):
    """查找实验的 best.pt，兼容路径嵌套。"""
    candidates = [
        SERVER_RUNS_BASE / exp_name / "weights" / "best.pt",
        LOCAL_RUNS_BASE / "detect" / exp_name / "weights" / "best.pt",
        LOCAL_RUNS_BASE / "ablation" / exp_name / "weights" / "best.pt",
        LOCAL_RUNS_BASE / "sa_nwd" / exp_name / "weights" / "best.pt",
        LOCAL_RUNS_BASE / "detect" / "runs" / "detect" / exp_name / "weights" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def eval_model(weights_path, name, data="configs/data.yaml", imgsz=1280, split="val"):
    """评估单个模型，返回完整指标。"""
    import register_modules  # noqa
    from ultralytics import YOLO
    import torch

    print(f"  Evaluating {name}...")
    model = YOLO(weights_path)
    metrics = model.val(data=data, imgsz=imgsz, split=split, verbose=False)

    # 精度指标
    map50 = round(float(metrics.box.map50), 4)
    map75 = round(float(metrics.box.map75), 4)
    map = round(float(metrics.box.map), 4)
    precision = round(float(metrics.box.mp), 4)
    recall = round(float(metrics.box.mr), 4)
    f1 = round(2 * precision * recall / (precision + recall + 1e-8), 4)

    # 效率指标
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # FLOPs
    try:
        from ultralytics.utils.torch_utils import get_flops
        flops = get_flops(model.model, imgsz) / 1e9
    except Exception:
        flops = 0.0

    # 模型文��大小
    size_mb = Path(weights_path).stat().st_size / 1e6

    # FPS（简单估算）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import time
        dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
        model.model.to(device).eval()
        with torch.no_grad():
            for _ in range(10):  # warmup
                model.model(dummy)
            t0 = time.perf_counter()
            for _ in range(50):
                model.model(dummy)
            fps = 50 / (time.perf_counter() - t0)
    except Exception:
        fps = 0.0

    return {
        "name": name,
        "weights": weights_path,
        "map50": map50,
        "map75": map75,
        "map50_95": map,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "params_M": round(params, 3),
        "flops_G": round(flops, 2),
        "fps": round(fps, 1),
        "size_mb": round(size_mb, 2),
    }


def format_ablation_table(results):
    """格式化消融实验表格。"""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study Results on UAV Dataset}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{mAP$_{50}$} & \\textbf{mAP$_{50:95}$} & \\textbf{Params(M)} & \\textbf{$\\Delta$mAP$_{50}$} \\\\")
    lines.append("\\midrule")

    baseline_map50 = None
    for r in results:
        if "baseline" in r["name"].lower() or r["name"] == "Baseline":
            baseline_map50 = r["map50"]

    for r in results:
        delta = ""
        if baseline_map50 and r["name"] != "Baseline":
            d = r["map50"] - baseline_map50
            delta = f"+{d*100:.1f}" if d >= 0 else f"{d*100:.1f}"

        is_best = r == max(results, key=lambda x: x["map50"])
        fmt = "\\textbf{" if is_best else ""
        end = "}" if is_best else ""
        lines.append(
            f"  {fmt}{r['name']}{end} & {fmt}{r['map50']*100:.1f}{end} & "
            f"{fmt}{r['map50_95']*100:.1f}{end} & {r['params_M']:.2f} & {delta} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def format_efficiency_table(results):
    """格式化效率对比表格。"""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Efficiency Comparison}")
    lines.append("\\label{tab:efficiency}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Params(M)} & \\textbf{FLOPs(G)} & \\textbf{FPS} & \\textbf{Size(MB)} & \\textbf{mAP$_{50}$} \\\\")
    lines.append("\\midrule")
    for r in results:
        lines.append(
            f"  {r['name']} & {r['params_M']:.2f} & {r['flops_G']:.1f} & "
            f"{r['fps']:.0f} & {r['size_mb']:.1f} & {r['map50']*100:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    # 定义要评估的实验
    EXPERIMENTS = [
        # 消融实验（标准 NWD）
        ("Baseline", "ablation/baseline"),
        ("+NWD", "ablation/nwd_only"),
        ("+NWD+P2", "ablation/nwd_p2"),
        ("+NWD+P2+SimAM", "ablation/nwd_p2_simam"),
        ("+NWD+P2+SimAM+PConv", "ablation/nwd_p2_simam_pconv"),
        ("Full (NWD)", "ablation/full_improved"),
        # SA-NWD 两阶段实验
        ("Improved Two-Stage", "two_stage_s2"),
        ("SOTA Two-Stage", "sota_s2"),
    ]

    print("=" * 70)
    print("  Collecting All Experiment Results")
    print("=" * 70)

    all_results = []
    missing = []

    for name, exp_name in EXPERIMENTS:
        weights = find_best_pt(exp_name)
        if weights is None:
            print(f"  [SKIP] {name}: no weights found for '{exp_name}'")
            missing.append(name)
            continue
        try:
            r = eval_model(weights, name)
            all_results.append(r)
            print(f"  {name}: mAP50={r['map50']:.4f}, mAP50-95={r['map50_95']:.4f}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            missing.append(name)

    if not all_results:
        print("No results collected. Check experiment paths.")
        return

    # Save JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {'Method':<30} {'mAP50':>7} {'mAP75':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*65}")
    for r in all_results:
        print(f"  {r['name']:<30} {r['map50']*100:>7.1f} {r['map75']*100:>7.1f} "
              f"{r['map50_95']*100:>9.1f} {r['precision']*100:>7.1f} "
              f"{r['recall']*100:>7.1f} {r['f1']*100:>7.1f}")
    print(f"  {'-'*65}")
    print(f"  {'Method':<30} {'Params':>8} {'FLOPs':>7} {'FPS':>6} {'Size':>7}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['name']:<30} {r['params_M']:>7.2f}M {r['flops_G']:>6.1f}G "
              f"{r['fps']:>5.0f} {r['size_mb']:>6.1f}MB")
    print(f"{'='*70}")

    # Generate LaTeX tables
    ablation_results = [r for r in all_results
                        if any(kw in r['name'] for kw in ["Baseline", "NWD", "P2", "SimAM", "PConv", "Full"])]
    main_results = [r for r in all_results
                    if any(kw in r['name'] for kw in ["Baseline", "Two-Stage", "SOTA"])]

    latex = "% Auto-generated by collect_results.py\n\n"
    if ablation_results:
        latex += format_ablation_table(ablation_results) + "\n\n"
    if main_results:
        latex += format_efficiency_table(main_results)

    with open("outputs/latex_tables.tex", "w") as f:
        f.write(latex)

    print(f"\nOutputs saved to outputs/")
    if missing:
        print(f"Missing experiments: {missing}")


if __name__ == "__main__":
    main()
