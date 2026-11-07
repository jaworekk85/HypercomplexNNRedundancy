"""Read-only synthesis of completed QNN compression evidence, with limits."""

from pathlib import Path

import numpy as np

from scripts.compression_curve import read
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]


def run():
    output=ROOT / "results/qnn_compression_review";output.mkdir(parents=True,exist_ok=True)
    tables=[];effects=[];inputs={};pending=[]
    for dataset,suffix in (("cifar",""),("eurosat","_eurosat")):
        for extension in ("","_refinement"):
            folder=ROOT / f"results/compression_curve{suffix}{extension}";path=folder / "summary.json"
            if not path.exists() or read(path)["status"]!="completed":
                pending.append(str(path.relative_to(ROOT)));continue
            data=read(path);inputs[str(path.relative_to(ROOT))]=file_sha256(path)
            lookup={(r["architecture"],r["fraction"],r["policy"],r["stage"]):r for r in data["aggregates"]}
            seeds=data["protocol"]["config"]["seeds"]
            base=lookup["cnn6",0.,"baseline","before"]
            for fraction in data["protocol"]["config"]["fractions"]:
                row=lookup["cnn6",fraction,"greedy_comp","after"]
                val=lambda architecture,policy,stage:lookup.get((architecture,fraction,policy,stage))
                accuracy=lambda r:100*r["validation"]["mean_accuracy"] if r else None
                checkpoint=folder / f"compact/cnn6/seed_{seeds[0]}/f_{round(1000*fraction):03d}/greedy_comp.pt"
                tables.append({"dataset":dataset,"followup_refinement":bool(extension),"fraction_removed_at_each_site":fraction,
                    "actual_real_parameters":row["real_parameters"],"parameter_savings_percent":100*(1-row["real_parameters"]/base["real_parameters"]),
                    "seed_first_inference_checkpoint_bytes":checkpoint.stat().st_size if checkpoint.exists() else None,
                    "baseline_before_accuracy":accuracy(base),"comp_before_accuracy":accuracy(val("cnn6","greedy_comp","before")),
                    "comp_after_accuracy":accuracy(row),"bn_after_accuracy":accuracy(val("cnn6","bn_pruned","after")),
                    "random_after_accuracy":accuracy(val("cnn6","random","after")),"scratch_accuracy":accuracy(val("cnn6","scratch","scratch")),
                    "real_comp_after_accuracy":accuracy(val("real_cnn6","greedy_comp","after"))})
                for left,right,stage in (("greedy_comp","greedy_pruned","before"),("greedy_comp","bn_pruned","after"),("greedy_comp","random","after"),
                                         ("greedy_comp","greedy_full_projected","before"),("greedy_comp","scratch","after")):
                    a=val("cnn6",left,stage);b=val("cnn6",right,"scratch" if right=="scratch" else stage)
                    if not a or not b: continue
                    av={r["seed"]:r["accuracy"] for r in a["validation"]["seed_values"]};bv={r["seed"]:r["accuracy"] for r in b["validation"]["seed_values"]}
                    differences=[100*(av[s]-bv[s]) for s in seeds]
                    effects.append({"dataset":dataset,"fraction":fraction,"left":left,"right":right,"left_stage":stage,
                        "right_stage":"scratch" if right=="scratch" else stage,"seeds":seeds,"seed_pp":differences,"mean_pp":float(np.mean(differences)),
                        "positive_seeds":sum(v>0 for v in differences),"unequal_student_training_budget":right=="scratch"})
    oracle=[]
    for dataset,name in (("cifar","results/multisource_compensation/summary.json"),("eurosat","results/eurosat_compression/summary.json")):
        path=ROOT / name;data=read(path);inputs[name]=file_sha256(path)
        aggregates=data["phase_a"]["aggregates"] if dataset=="eurosat" else data["aggregates"]
        for removed in (4,8):
            selected={r["policy"]:r for r in aggregates if r["architecture"]=="cnn6" and r["removed"]==removed}
            q,full=(selected[p]["validation"] for p in ("multi_q","multi_full_substitution"))
            qv={r["seed"]:r["accuracy"] for r in q["seed_values"]};fv={r["seed"]:r["accuracy"] for r in full["seed_values"]}
            seeds=sorted(set(qv)&set(fv));differences=[100*(qv[s]-fv[s]) for s in seeds]
            oracle.append({"dataset":dataset,"site":"block-2 output only","removed_of_16":removed,
                           "compact_q_accuracy":100*q["mean_accuracy"],"functional_full_accuracy":100*full["mean_accuracy"],
                           "seeds":seeds,"q_minus_functional_full_seed_pp":differences,"mean_pp":float(np.mean(differences)),
                           "functional_full_is_compact_qnn":False})
    summary={"status":"completed_for_available_finished_curves","input_hashes":inputs,"pending_curves":pending,"rows":tables,"paired_effects":effects,"oracle_diagnostics":oracle,
             "new_training_performed":False,"official_test_evaluated":False}
    write_json(output / "summary.json",summary)
    lines=["# Compression of an existing QNN family: evidence and limits","","## Hypotheses","",
        "Characterize when Hamilton-structure-preserving compensation retains useful quality, its actual compact size and failure regimes. No novelty or intrinsic quaternion superiority is assumed. A separate matched circulant control tests one alternative weight-sharing pattern; pseudo-label repair is supplementary.","",
        "## Measured results","",
        "This report reads completed studies only and performs no new training or inference. Three seeds per dataset. Uniform removal at every convolution output; physical narrower networks. Random trials are averaged within seed before aggregation. Refinement points were selected after coarse outcomes and retain that follow-up status.","",
        "| Dataset | Removed/site | Real params | Parameter saving | Comp before | Comp +3 ep | BN +3 ep | Scratch 50 ep | Real comp +3 ep |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt=lambda v:f"{v:.2f}%" if v is not None else "pending"
    for r in sorted(tables,key=lambda x:(x["dataset"],x["fraction_removed_at_each_site"])):
        lines.append(f"| {r['dataset']} | {r['fraction_removed_at_each_site']} | {r['actual_real_parameters']} | {fmt(r['parameter_savings_percent'])} | {fmt(r['comp_before_accuracy'])} | {fmt(r['comp_after_accuracy'])} | {fmt(r['bn_after_accuracy'])} | {fmt(r['scratch_accuracy'])} | {fmt(r['real_comp_after_accuracy'])} |")
    lines.extend(["","Paired compensation minus BN after equal three-epoch repair (validation pp; unfavorable seeds retained):",""])
    for r in effects:
        if r["right"]=="bn_pruned": lines.append(f"- {r['dataset']}, fraction {r['fraction']}: {', '.join(f'{v:+.2f}' for v in r['seed_pp'])}; mean {r['mean_pp']:+.2f} pp.")
    lines.extend(["","Earlier single-site comparison: compact structure-preserving QNN versus unrestricted functional substitution, without repair:","",
                  "| Dataset | Removed at block-2 output | Compact Q accuracy | Functional unrestricted accuracy | Mean Q minus unrestricted (pp) |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for r in oracle: lines.append(f"| {r['dataset']} | {r['removed_of_16']}/16 | {r['compact_q_accuracy']:.2f}% | {r['functional_full_accuracy']:.2f}% | {r['mean_pp']:+.2f} |")
    lines.extend(["","The unrestricted functional reference retains extra computation and is NOT a compact QNN at the same budget. It is a representation diagnostic; this table is a different experiment from all-site compression above.","",
        "## Interpretation and limits","",
        "Half the units per site is about three quarters of parameters removed, not half the parameters. Parameter sharing and actual compact architecture savings are distinct; a mask alone is not compression. Stored checkpoint bytes depend on serialization and are not latency or MAC measurements.","",
        "Scratch models get 50 student-training epochs; repair gets three in addition to a previously trained teacher. Their comparison describes this resource allocation, not equal-compute superiority. Fully trained Q and real baselines may differ in quality; compare their own-baseline losses as well as absolute scores.","",
        "Directly constrained fitting versus full fit followed by projection compares fitting strategies at the same compact architecture. It is not the cost relative to an unconstrained functional substitution oracle; such an oracle is not a compact QNN. Report the earlier block-2 unrestricted substitution study separately (results/multisource_compensation/REPORT.md, results/eurosat_compression/REPORT.md). Prior context: GRAIL2026, Kim2020, Asiaee2026CMR; the four-coefficient restriction is not a new algorithm.","",
        "The strong-compression failure regimes and any negative paired effects above are evidence, not runs to discard. A benefit over pruning does not establish useful absolute accuracy, a universal advantage after repair, or a general advantage of quaternions. Three seeds and reused validation data constrain generalization claims. CIFAR test was used in early pilots; no new test inference here. EuroSAT test remains held out.","",
        "Pending/incomplete curves are omitted rather than treated as findings: "+(", ".join(pending) or "none"),"",
        "Original plots/reports: results/compression_curve and results/compression_curve_eurosat. New structural control: configs/structure_control.json. Exact per-seed effects and source hashes: summary.json."])
    (output / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Reviewed {len(inputs)} completed input studies; {len(tables)} QNN operating points. No new training or inference.",flush=True)


if __name__=="__main__": run()
