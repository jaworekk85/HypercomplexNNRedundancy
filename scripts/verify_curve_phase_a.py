"""Audit the completed no-fine-tuning curve, its saved compact states and inputs."""

from pathlib import Path
import sys

import numpy as np
import torch

from scripts.compression_curve import factory,kind_of,read,finetuned
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]


def verify(output):
    output=Path(output);summary=read(output / "summary.json");config=summary["protocol"]["config"]
    manifest=read(output / "input_manifest.json");checks={}

    def check(name,value):
        checks[name]=bool(value)

    for name,digest in {**summary["protocol"]["source_hashes"],**manifest["hashes"]}.items():
        check(f"preserved:{name}",file_sha256(ROOT / name)==digest)
    check("same_config",manifest["config"]==config)
    check("no_new_official_test",not summary["official_test_evaluated"] and not config["official_test_evaluation"])
    rows=[r for r in summary["rows"] if r["stage"]=="before"]
    expected=sum(1+len(config["fractions"])*(3+(arch=="cnn6")+2*config["random_repetitions"]) for arch in config["architectures"])*len(config["seeds"])
    check("all_no_finetuning_rows_present",len(rows)==expected)
    baseline_sources={(r["architecture"],r["seed"]):r for r in manifest["baselines"]}
    counts={}
    for arch in config["architectures"]:
        for seed in config["seeds"]:
            key=f"{arch}:{seed}";source=ROOT / baseline_sources[(arch,seed)]["path"];kind=kind_of(arch)
            original=read(source / "summary.json")["config"]
            widths=original["quaternion_widths" if kind=="quaternion" else "real_widths"]
            with np.load(source / "split_indices.npz") as saved:
                ids=[saved[name] for name in ("train","validation","calibration")]
                check(f"{key}:disjoint_splits",len(np.unique(np.concatenate(ids)))==sum(map(len,ids)))
                check(f"{key}:whole_image_fit_check",len(saved["calibration"][::2])==len(saved["calibration"][1::2])==500 and len(saved["validation"])==1000)
            saved=read(output / "phase_a" / f"{arch}_seed_{seed}.json")
            check(f"{key}:source_rows_preserved",saved["rows"]==[r for r in rows if (r["architecture"],r["seed"])==(arch,seed)])
            records={(r["fraction"],r["case"]):r for r in saved["records"]}
            for row in saved["rows"]:
                tag=f"{key}:{row['fraction']}:{row['case']}";fraction=row["fraction"]
                for split in ("validation","check_calibration"):
                    metrics=row["evaluation"][split]
                    check(f"{tag}:{split}:finite_metrics",all(np.isfinite(v) for v in metrics.values()) and 0<=metrics["accuracy"]<=1 and metrics["cross_entropy"]>=0)
                if fraction:
                    record=records[(fraction,row["case"])]
                    check(f"{tag}:six_physical_sites",[r["site"] for r in record["sites"]]==[[block,layer] for block in range(3) for layer in (0,1)])
                    valid=True
                    for index,site in enumerate(record["sites"]):
                        width=widths[index//2];victims=site["victims"]
                        valid &= len(victims)==len(set(victims))==round(width*fraction) and min(victims)>=0 and max(victims)<width and site["kept"]==width-len(victims)
                        if "fold_error" in site:
                            valid &= site["fold_error"]<=config["equivalence_tolerance"]
                    check(f"{tag}:declared_removal_and_folds",valid)
                    if row["policy"] in ("greedy_pruned","greedy_full_projected"):
                        check(f"{tag}:same_greedy_victims",[s["victims"] for s in record["sites"]]==[s["victims"] for s in records[(fraction,"greedy_comp")]["sites"]])
                    if row["policy"]=="random_comp":
                        paired=f"random_r{row['random_repetition']:02d}"
                        check(f"{tag}:same_random_victims",[s["victims"] for s in record["sites"]]==[s["victims"] for s in records[(fraction,paired)]["sites"]])
                if row["policy"]=="baseline" or finetuned(config,row["policy"],row["random_repetition"]):
                    path=output / "compact" / arch / f"seed_{seed}" / f"f_{round(1000*fraction):03d}" / f"{row['case']}.pt"
                    state=torch.load(path,map_location="cpu",weights_only=True)["model"]
                    compact={**original};width_key="quaternion_widths" if kind=="quaternion" else "real_widths"
                    compact[width_key]=[w-round(w*fraction) for w in widths]
                    model=factory(config["dataset"],arch)(compact,kind)
                    model.load_state_dict(state)
                    real_count=sum(p.numel() for p in model.parameters())
                    check(f"{tag}:actual_narrow_checkpoint_budget",real_count==row["real_parameters"] and all(model.blocks[b][3*layer].out_channels==compact[width_key][b] for b in range(3) for layer in (0,1)))
                    counts[tag]={"parameters":real_count,"buffers":sum(b.numel() for b in model.buffers()),"bytes":path.stat().st_size}
    for item in [r for r in summary["aggregates"] if r["stage"]=="before"]:
        tag=f"aggregate:{item['architecture']}:{item['fraction']}:{item['policy']}"
        selected=[r for r in rows if (r["architecture"],r["fraction"],r["policy"])==(item["architecture"],item["fraction"],item["policy"])]
        for split in ("validation","check_calibration"):
            values=[np.mean([r["evaluation"][split]["accuracy"] for r in selected if r["seed"]==seed]) for seed in config["seeds"]]
            check(f"{tag}:{split}:seed_mean_and_sd",len(item[split]["seed_values"])==3 and abs(np.mean(values)-item[split]["mean_accuracy"])<1e-12 and abs(np.std(values,ddof=1)-item[split]["sample_sd_accuracy"])<1e-12)
    result={"status":"passed" if all(checks.values()) else "failed","scope":"Phase A only; no verification claim for unfinished fine-tuning/scratch phases",
            "checks":checks,"saved_compact_state_counts":counts,"verifier_sha256":file_sha256(Path(__file__)),"official_test_evaluated":False}
    write_json(output / "phase_a_verification.json",result)
    failed=[name for name,value in checks.items() if not value]
    print({"status":result["status"],"saved_compact_models":len(counts),"failed":failed},flush=True)
    if failed: raise AssertionError("Curve phase-A artifact checks failed")
    return result


if __name__=="__main__":
    verify(ROOT / (sys.argv[1] if len(sys.argv)>1 else "results/compression_curve"))
