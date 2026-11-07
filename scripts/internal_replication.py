"""Replicate intermediate compression across two depths and removal levels."""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import make_model as make_cnn6,write_json
from scripts.internal_merging import cache_path,consumer_errors
from src.cnn_training import evaluate,prepare_batch,seed_everything
from src.compression.internal_merging import fit_pairs,grouped_to_tensor,outgoing_addition,select_pairs,substitute_features
from src.compression.quaternion_channels import compact_block
from src.compression.weighted_projection import add_weighted_maps,compact_policy,weighted_pairs
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.evaluation.block_ablation import capture_blocks,classification_metrics,predict_from_block
from src.models.cifar_fourblock import make_fourblock

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT / ".cache/matplotlib"))
SOURCES=["scripts/internal_replication.py","src/compression/weighted_projection.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_model(source):
    return make_fourblock(source["config"]) if source["architecture"]=="cnn8" else make_cnn6(source["config"],"quaternion")


def sources_for(study):
    values=[]
    for row in read(ROOT / study["source_architecture_a"] / "summary.json")["runs"]:
        source=ROOT / row["path"]
        values.append({"architecture":"cnn6","seed":row["seed"],"path":str(source.relative_to(ROOT)).replace("\\","/"),
                       "config":read(source / "summary.json")["config"],"cache_path":str(cache_path(row["seed"]).relative_to(ROOT)).replace("\\","/")})
    values.extend(read(ROOT / study["output"] / "architecture_b/sources.json")["sources"])
    return values


def load_case(manifest,device):
    model=compact_block(make_model(manifest["source"]),1,manifest["kept_units"]).to(device).eval()
    model.load_state_dict(torch.load(ROOT / manifest["checkpoint_path"],map_location=device,weights_only=True)["model"])
    return model


def render(output,summary):
    config=summary["protocol"]["config"]
    policies=["baseline","random","paired_pruned","quaternion_exact","full_projected","weighted_projected","full_substitution"]
    aggregates=[]
    for architecture in ("cnn6","cnn8"):
        for removed in config["removed_units"]:
            for policy in policies:
                seed_values=[];check_seed_values=[]
                for seed in config["seeds"]:
                    rows=[row for row in summary["rows"] if row["architecture"]==architecture and row["removed"]==removed and row["seed"]==seed and row["policy"]==policy]
                    seed_values.append({"seed":seed,"accuracy":float(np.mean([row["evaluation"]["validation"]["accuracy"] for row in rows])),
                                        "cross_entropy":float(np.mean([row["evaluation"]["validation"]["cross_entropy"] for row in rows]))})
                    check_seed_values.append({"seed":seed,"accuracy":float(np.mean([row["evaluation"]["check_calibration"]["accuracy"] for row in rows])),
                                              "cross_entropy":float(np.mean([row["evaluation"]["check_calibration"]["cross_entropy"] for row in rows]))})
                sample=next(row for row in summary["rows"] if row["architecture"]==architecture and row["removed"]==removed and row["policy"]==policy)
                aggregates.append({"architecture":architecture,"removed":removed,"policy":policy,"seed_values":seed_values,
                                   "real_parameters":sample["real_parameters"],"mean_accuracy":float(np.mean([row["accuracy"] for row in seed_values])),
                                   "sample_sd_accuracy":float(np.std([row["accuracy"] for row in seed_values],ddof=1)),
                                   "mean_cross_entropy":float(np.mean([row["cross_entropy"] for row in seed_values])),
                                   "check_seed_values":check_seed_values,"check_mean_accuracy":float(np.mean([row["accuracy"] for row in check_seed_values])),
                                   "check_sample_sd_accuracy":float(np.std([row["accuracy"] for row in check_seed_values],ddof=1)),
                                   "check_mean_cross_entropy":float(np.mean([row["cross_entropy"] for row in check_seed_values]))})
    write_json(output / "aggregated.json",aggregates)
    effects=[]
    for architecture in ("cnn6","cnn8"):
        for removed in config["removed_units"]:
            lookup={row["policy"]:row for row in aggregates if row["architecture"]==architecture and row["removed"]==removed}
            for left,right in (("quaternion_exact","full_projected"),("quaternion_exact","random"),("quaternion_exact","paired_pruned"),("weighted_projected","quaternion_exact"),("full_substitution","full_projected")):
                differences=[100*(a["accuracy"]-b["accuracy"]) for a,b in zip(lookup[left]["seed_values"],lookup[right]["seed_values"])]
                ce_differences=[a["cross_entropy"]-b["cross_entropy"] for a,b in zip(lookup[left]["seed_values"],lookup[right]["seed_values"])]
                effects.append({"architecture":architecture,"removed":removed,"left":left,"right":right,"seed_differences_pp":differences,
                                "mean_pp":float(np.mean(differences)),"sample_sd_pp":float(np.std(differences,ddof=1)),
                                "seed_ce_differences":ce_differences,"mean_ce_difference":float(np.mean(ce_differences))})
    write_json(output / "paired_effects.json",effects)
    rows=["# Internal merging: replication across CNN depths and compression levels","","## Hypotheses and prior outcome","",
          "The cost of restricting full-map compensation to Hamilton blocks may persist across removal levels and CNN depths. "
          "Direct constrained fitting and data-weighted projection are equivalent; their agreement is a strong control, "
          "not a new algorithm. The earlier CNN6/four-removal cell is reused exploratory evidence, not independent replication.","",
          "## Prespecified protocol","",
          "CNN6: three existing 50-epoch baselines, widths 8/16/32. CNN8: three new 50-epoch baselines, "
          "four blocks with widths 8/16/24/32. Both are sequential BN/ReLU/max-pooling CNNs; architecture diversity "
          "is limited to depth/width within that family. Seeds 2026/2027/2028, same CIFAR split, normalization, "
          "augmentation, AdamW and batch/epoch budget. Choose baseline checkpoint by validation CE. "
          "No official-test inference; validation was used for checkpoint selection and is not a new blind benchmark.","",
          "Architectures differ in parameter count and compute; equal epoch budgets do not make them matched-cost "
          "baselines. Compression comparisons are paired within each architecture and training seed.","",
          "Fit zero-intercept maps on 500 even calibration images, check on 500 odd images. Use nested prefixes "
          "of four/eight common disjoint fit-only pairs, selected by full-map residual times outgoing kernel energy. "
          "Freeze all manifests before compressed-model validation. Compare paired pruning, direct-Q folding, "
          "unweighted projection, data-moment projection, and ten nested random removals per condition. "
          "No compression optimizer updates or BN refit. Full substitution is a functional oracle without a "
          "compact-architecture claim. Zero-offset Q/full fits use four/sixteen real calibration coefficients per pair; "
          "these maps are folded into the compact network and are not extra inference parameters. "
          "Report actual budgets per architecture, not just removed-unit fractions.","",
          "## Completed new baseline trainings","",
          "| Architecture | Seed | Epochs | Selected epoch | Validation accuracy | Validation CE |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for source in summary["protocol"]["sources"]:
        if source["architecture"]=="cnn8":
            metrics=source["metrics"]
            rows.append(f"| cnn8 | {source['seed']} | 50 | {metrics['best_epoch']} | {100*metrics['selected_validation']['accuracy']:.2f}% | {metrics['selected_validation']['cross_entropy']:.6f} |")
    rows += ["",
          "## Measured validation results","",
          "| Architecture | Removed units | Policy | Real parameters | Accuracy mean ± seed SD | Mean CE |",
          "| --- | ---: | --- | ---: | ---: | ---: |"]
    for item in aggregates:
        params=str(item["real_parameters"]) if item["real_parameters"] is not None else "functional oracle"
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['policy']} | {params} | {100*item['mean_accuracy']:.2f}% ± {100*item['sample_sd_accuracy']:.2f} pp | {item['mean_cross_entropy']:.6f} |")
    rows += ["","Random repetitions are averaged within training seed before SD across three seeds.","",
             "## Secondary classification check on 500 held-out calibration images","",
             "These images are disjoint from the 500 fit images and from validation. Check labels did not "
             "select the maps or compact policies. Structural consumer errors below also use this half.","",
             "| Architecture | Removed | Policy | Check accuracy mean ± seed SD | Check mean CE |",
             "| --- | ---: | --- | ---: | ---: |"]
    for item in aggregates:
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['policy']} | {100*item['check_mean_accuracy']:.2f}% ± {100*item['check_sample_sd_accuracy']:.2f} pp | {item['check_mean_cross_entropy']:.6f} |")
    rows += ["",
             "## Structural diagnostics and strong control","",
             "| Architecture | Removed | Seed | Discarded added-block energy | Unweighted/oracle consumer MSE | Direct-Q/oracle consumer MSE | Weighted/direct map error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summary["diagnostics"]:
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['seed']} | {100*item['discarded_full_energy_fraction']:.2f}% | {item['consumer_check']['full_projected_over_oracle_error_ratio']:.4f} | {item['quaternion_over_oracle_error_ratio']:.4f} | {item['weighted_map_max_error']:.3g} |")
    rows += ["","## Paired effects, including unfavorable seeds","",
             "| Architecture | Removed | Comparison (left minus right) | Accuracy per seed (pp) | Mean (pp) | CE per seed |",
             "| --- | ---: | --- | --- | ---: | --- |"]
    for item in effects:
        accuracy=", ".join(f"{value:+.2f}" for value in item["seed_differences_pp"])
        ce=", ".join(f"{value:+.6f}" for value in item["seed_ce_differences"])
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['left']} - {item['right']} | {accuracy} | {item['mean_pp']:+.2f} | {ce} |")
    rows += ["","## Interpretation and limits","",
             "Projection energy is the squared Frobenius norm of ADDED compensation blocks, not complete network weights. "
             "Consumer MSE, CE and classification accuracy are separate outcomes. The zero-intercept restriction "
             "avoids invalid constant-bias folding through zero padding. No map is commuted through ReLU. "
             "Exact-Q folding means equality with the graph after the chosen feature substitution, not "
             "equality with the original network: the fitted substitution itself is approximate. "
             "The functional oracle is a flexible feature-fit reference, not a task-accuracy upper bound. "
             "Observed consumer-error differences are not lower bounds on the best possible constrained "
             "fit: pairwise feature fitting does not optimize the joint consumer convolution. "
             "All positive and negative seed effects are retained. Data-weighted projection should reproduce direct-Q "
             "fits and predictions; any difference beyond tolerance is an integrity problem, not a gain. "
             "No new method, broad quaternion superiority or publication readiness is claimed. More datasets or a "
             "different architecture family would still be needed for broader generalization. Prior context: "
             "Kim2020, Asiaee2026CMR, Kobayashi2010, Nitta2002 in references.bib.",""]
    rows += ["## Focused prior-art positioning","",
             "Next-convolution compensation and feature preservation are established in "
             "[Kim2020, sections 3.1-3.3](https://proceedings.neurips.cc/paper_files/paper/2020/file/0678ca2eae02d542cc931e81b74de122-Paper.pdf). "
             "Our maps act after a complete block, so their use does not assert ReLU commutation.","",
             "Weighted affine fits and ordinary affine-consumer folding are explicit in "
             "[Asiaee2026CMR, Appendix A.5](https://arxiv.org/html/2602.24266v2). "
             "Its discussion of cross-unit coupling reinforces that our pair-ranking proxy is not exact "
             "joint consumer-error minimization. Retaining Hamilton blocks is a constraint of this "
             "experiment; this limited review does not establish a literature gap. See LITERATURE.md.",""]
    (output / "REPORT.md").write_text("\n".join(rows),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    x=np.arange(2)
    for ax,architecture in zip(axes,("cnn6","cnn8")):
        for policy,label in (("quaternion_exact","Direct Q"),("full_projected","Unweighted projection"),("weighted_projected","Data-weighted projection"),("random","Random")):
            values=[next(row for row in aggregates if row["architecture"]==architecture and row["removed"]==removed and row["policy"]==policy) for removed in config["removed_units"]]
            ax.errorbar(x,[100*row["mean_accuracy"] for row in values],yerr=[100*row["sample_sd_accuracy"] for row in values],marker="s" if policy=="weighted_projected" else "o",linestyle="--" if policy=="weighted_projected" else "-",fillstyle="none" if policy=="weighted_projected" else "full",capsize=3,label=label)
        baseline=next(row for row in aggregates if row["architecture"]==architecture and row["policy"]=="baseline")
        ax.axhline(100*baseline["mean_accuracy"],linestyle=":",color="gray",label="Uncompressed")
        ax.set_xticks(x,[f"{count}/16 units" for count in config["removed_units"]]);ax.grid(alpha=.2)
        ax.set_title(architecture.upper());ax.set_ylabel("Validation accuracy (%), mean ± seed SD");ax.legend(fontsize=8)
    fig.suptitle("Intermediate compression: two depths and two removal levels; no fine-tuning")
    fig.tight_layout();fig.savefig(output / "comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,architecture in zip(axes,("cnn6","cnn8")):
        for field,label in (("discarded_full_energy_fraction","Discarded added-block energy"),("consumer_penalty","Extra MSE: unweighted vs full oracle"),("direct_penalty","Extra MSE: direct Q vs full oracle")):
            values=[];deviations=[]
            for removed in config["removed_units"]:
                records=[row for row in summary["diagnostics"] if row["architecture"]==architecture and row["removed"]==removed]
                points=[100*(row["consumer_check"]["full_projected_over_oracle_error_ratio"]-1) if field=="consumer_penalty" else 100*(row["quaternion_over_oracle_error_ratio"]-1) if field=="direct_penalty" else 100*row[field] for row in records]
                values.append(float(np.mean(points)));deviations.append(float(np.std(points,ddof=1)))
            ax.errorbar(x,values,yerr=deviations,marker="o",capsize=3,label=label)
        ax.axhline(0,color="gray",linewidth=.8);ax.set_xticks(x,[f"{count}/16 units" for count in config["removed_units"]])
        ax.set_title(architecture.upper());ax.set_ylabel("Percent, mean ± seed SD");ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Observed cost of retaining quaternion blocks; distinct diagnostics")
    fig.tight_layout();fig.savefig(output / "structural_cost.png",dpi=180);plt.close(fig)
    return aggregates,effects


@torch.no_grad()
def run(study):
    output=ROOT / study["output"]
    if study["official_test_evaluation"] or study["compression_optimizer_updates"] or study["bn_recalibration"] or study["removed_units"]!=[4,8] or study["intercept"]:
        raise ValueError("Only the frozen two-level, no-fine-tuning/no-test extension is implemented")
    sources=sources_for(study)
    old=read(ROOT / study["source_initial_study"] / "protocol.json")
    training=read(output / "architecture_b/training_protocol.json")
    hashes={**old["source_hashes"],**training["source_hashes"],**{name:file_sha256(ROOT / name) for name in SOURCES}}
    for name,digest in {**old["source_hashes"],**training["source_hashes"]}.items():
        if file_sha256(ROOT / name)!=digest: raise ValueError(f"Frozen numerical source changed: {name}")
    paths=[output / "architecture_b/sources.json",ROOT / study["source_initial_study"] / "summary.json"]
    for source in sources:
        folder=ROOT / source["path"]
        paths.extend([folder / "quaternion/model.pt",folder / "quaternion/metrics.json",folder / "split_indices.npz",folder / "data.json",ROOT / source["cache_path"]])
        if source["architecture"]=="cnn6":
            paths.append(ROOT / study["source_initial_study"] / f"seed_{source['seed']}/all_pair_fits.json")
    protected={str(path.relative_to(ROOT)).replace("\\","/"):file_sha256(path) for path in paths}
    protocol={"config":study,"source_hashes":hashes,"protected_inputs":protected,"sources":sources,
              "original_cell_is_not_independent_replication":True,"compression_optimizer_updates":0,"official_test_evaluated":False}
    if (output / "study_protocol.json").exists() and read(output / "study_protocol.json")!=protocol:
        raise ValueError("Replication sources/protocol changed; preserve the run and use another output")
    write_json(output / "study_protocol.json",protocol)
    if (output / "summary.json").exists():
        summary=read(output / "summary.json");render(output,summary);return summary
    torch.set_num_threads(study["cpu_threads"]);seed_everything(study["seeds"][0]);device=torch.device(study["device"])
    (images,labels),unused_test=load_cached_cifar(ROOT / study["archive"]);del unused_test
    stats=read(ROOT / study["data_source"] / "data.json")["statistics"]
    mean=torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    std=torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    manifests=[];selected={};audits=[]
    for source in sources:
        architecture,seed=source["architecture"],source["seed"]
        model=make_model(source).to(device).eval()
        state=torch.load(ROOT / source["path"] / "quaternion/model.pt",map_location=device,weights_only=True)["model"];model.load_state_dict(state)
        with np.load(ROOT / source["cache_path"]) as cached: values=cached["trained"]
        if values.shape!=(1000,64,16,4): raise ValueError("Unexpected intermediate feature layout")
        folder=output / "conditions" / architecture / f"seed_{seed}"
        consumer=model.blocks[2][0].kernel.detach().cpu().double().numpy()
        if architecture=="cnn6":
            candidates=read(ROOT / study["source_initial_study"] / f"seed_{seed}/all_pair_fits.json")
        else:
            candidates=fit_pairs(values[::2],consumer)
            write_json(folder / "all_pair_fits.json",candidates)
        ordered=add_weighted_maps(select_pairs(candidates,8),values[::2])
        maximum=max(row["maps"]["weighted_projected"]["maximum_difference_from_direct"] for row in ordered)
        if maximum>study["weighted_projection_tolerance"]: raise AssertionError("Strong weighted projection control differs from direct Q fit")
        audits.append({"architecture":architecture,"seed":seed,"maximum_map_error":maximum})
        with np.load(ROOT / source["path"] / "split_indices.npz") as split: cal_ids=split["calibration"]
        probe_positions=np.arange(0,64,2)
        normalized=prepare_batch(torch.as_tensor(images[cal_ids[probe_positions]],device=device),mean,std)
        features=capture_blocks(model,normalized,32)[1]
        if float((features-grouped_to_tensor(values[probe_positions])).abs().max())>study["equivalence_tolerance"]: raise AssertionError("Source hidden cache mismatch")
        baseline_path=folder / "baseline_standardized.pt"
        baseline_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"model":{name:value.detach().cpu() for name,value in model.state_dict().items()}},baseline_path)
        for removed in study["removed_units"]:
            pairs=deepcopy(ordered[:removed]);key=f"{architecture}:{seed}:{removed}";selected[key]=pairs
            case_base=folder / f"removed_{removed:02d}"
            specs=[("baseline","baseline",None)]
            specs.extend((policy,policy,None) for policy in ("paired_pruned","quaternion_exact","full_projected","weighted_projected"))
            specs.extend((f"random_r{rep:02d}","random",rep) for rep in range(study["random_repetitions"]))
            for case_name,policy,rep in specs:
                case_folder=case_base / case_name
                if policy=="baseline": compact,kept,path=deepcopy(model),list(range(16)),baseline_path
                elif policy=="random":
                    victim=np.random.default_rng(study["random_seed_offset"]+seed*100+rep).permutation(16)[:removed]
                    kept=sorted(set(range(16))-set(victim));compact=compact_block(model,1,kept).eval();path=case_folder / "model.pt"
                else:
                    compact,kept,_=compact_policy(model,pairs,policy);path=case_folder / "model.pt"
                if policy!="baseline":
                    path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save({"model":{name:value.detach().cpu() for name,value in compact.state_dict().items()}},path)
                if policy in ("quaternion_exact","weighted_projected"):
                    expected=predict_from_block(model,1,substitute_features(features.to(device),weighted_pairs(pairs) if policy=="weighted_projected" else pairs,"quaternion_right"),batch_size=32)
                else:
                    reference=deepcopy(model)
                    if policy=="full_projected":
                        for pair in pairs:
                            addition,_=outgoing_addition(consumer,pair["victim"],pair["maps"]["full_real"]["matrix"])
                            target=reference.blocks[2][0].kernel[:,:,pair["keeper"]];target.add_(torch.as_tensor(addition,dtype=target.dtype,device=device))
                    expected=predict_from_block(reference,1,features,sorted(set(range(16))-set(kept)),32)
                error=float(np.max(np.abs(compact(normalized).cpu().numpy()-expected)))
                if error>study["equivalence_tolerance"]: raise AssertionError(f"Graph/fold check failed: {key}/{case_name}: {error}")
                manifest={"architecture":architecture,"seed":seed,"removed":removed,"case":case_name,"policy":policy,"source":source,
                          "kept_units":kept,"pairs":pairs,"random_repetition":rep,"probe_error":error,
                          "checkpoint_path":str(path.relative_to(ROOT)).replace("\\","/"),"checkpoint_sha256":file_sha256(path),
                          "reused_exploratory_condition":architecture=="cnn6" and removed==4}
                write_json(case_folder / "manifest.json",manifest);manifests.append(manifest)
            print(f"{architecture} seed {seed}: {removed}/16 removal, all choices frozen",flush=True)
    write_json(output / "selection_frozen.json",{"manifests":manifests,"pairs":selected,"weighted_map_audits":audits,"all_choices_frozen_before_compressed_validation":True})
    rows=[];diagnostics=[]
    for source in sources:
        architecture,seed=source["architecture"],source["seed"]
        original=make_model(source).to(device).eval();original.load_state_dict(torch.load(ROOT / source["path"] / "quaternion/model.pt",map_location=device,weights_only=True)["model"])
        with np.load(ROOT / source["path"] / "split_indices.npz") as split: ids={"check_calibration":split["calibration"][1::2],"validation":split["validation"]}
        with np.load(ROOT / source["cache_path"]) as cached: check_features=grouped_to_tensor(cached["trained"][1::2])
        val_features=capture_blocks(original,prepare_batch(torch.as_tensor(images[ids["validation"]],device=device),mean,std),study["batch_size"])[1]
        for removed in study["removed_units"]:
            key=f"{architecture}:{seed}:{removed}";pairs=selected[key];models={};case_base=output / "conditions" / architecture / f"seed_{seed}/removed_{removed:02d}"
            for manifest in [item for item in manifests if item["architecture"]==architecture and item["seed"]==seed and item["removed"]==removed]:
                model=load_case(manifest,device);metrics={};predictions={}
                for split,split_ids in ids.items():
                    metrics[split],predictions[f"{split}_logits"]=evaluate(model,torch.as_tensor(images[split_ids],device=device),torch.as_tensor(labels[split_ids],device=device),mean,std,study["batch_size"])
                    predictions[f"{split}_labels"]=labels[split_ids];predictions[f"{split}_ids"]=split_ids
                folder=case_base / manifest["case"]
                np.savez_compressed(folder / "predictions.npz",**predictions)
                result={"architecture":architecture,"seed":seed,"removed":removed,"case":manifest["case"],"policy":manifest["policy"],"physical":True,
                        "masked_parameters":0,"additional_inference_map_parameters":0,
                        "real_parameters":sum(p.numel() for p in model.parameters()),"buffer_elements":sum(b.numel() for b in model.buffers()),
                        "checkpoint_bytes":(ROOT / manifest["checkpoint_path"]).stat().st_size,"evaluation":metrics,
                        "manifest_sha256":file_sha256(folder / "manifest.json"),"predictions_sha256":file_sha256(folder / "predictions.npz")}
                write_json(folder / "metrics.json",result);rows.append(result)
                if manifest["policy"] in ("quaternion_exact","full_projected","paired_pruned","weighted_projected"): models[manifest["policy"]]=model
            errors=consumer_errors(original,models,check_features,pairs,study["batch_size"])
            full_deltas=[outgoing_addition(original.blocks[2][0].kernel.detach().cpu().double().numpy(),pair["victim"],pair["maps"]["full_real"]["matrix"])[1] for pair in pairs]
            diagnostic={"architecture":architecture,"seed":seed,"removed":removed,"consumer_check":errors,
                        "quaternion_over_oracle_error_ratio":errors["quaternion_error"]/errors["full_oracle_error"],
                        "discarded_full_energy_fraction":sum(item["discarded_squared_norm"] for item in full_deltas)/sum(item["full_squared_norm"] for item in full_deltas),
                        "weighted_map_max_error":max(pair["maps"]["weighted_projected"]["maximum_difference_from_direct"] for pair in pairs)}
            if errors["quaternion_closure_max_abs_error"]>study["equivalence_tolerance"]: raise AssertionError("Exact-Q consumer folding failed")
            for split in ids:
                with np.load(case_base / "weighted_projected/predictions.npz") as weighted,np.load(case_base / "quaternion_exact/predictions.npz") as direct:
                    error=float(np.max(np.abs(weighted[f"{split}_logits"]-direct[f"{split}_logits"])))
                diagnostic[f"{split}_weighted_logit_error"]=error
                if error>study["equivalence_tolerance"]: raise AssertionError("Equivalent strong-control predictions differ")
            diagnostics.append(diagnostic);write_json(case_base / "consumer_diagnostics.json",diagnostic)
            for mode,policy in (("quaternion_right","quaternion_substitution"),("full_real","full_substitution")):
                metrics={};predictions={}
                for split,features in (("check_calibration",check_features),("validation",val_features)):
                    logits=predict_from_block(original,1,substitute_features(features.to(device),pairs,mode),batch_size=study["batch_size"])
                    metrics[split]=classification_metrics(logits,labels[ids[split]])
                    predictions[f"{split}_logits"]=logits;predictions[f"{split}_labels"]=labels[ids[split]];predictions[f"{split}_ids"]=ids[split]
                    if mode=="quaternion_right":
                        with np.load(case_base / "quaternion_exact/predictions.npz") as saved:
                            if np.max(np.abs(logits-saved[f"{split}_logits"]))>study["equivalence_tolerance"]: raise AssertionError("Physical Q differs from functional substitution")
                folder=case_base / policy;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder / "predictions.npz",**predictions)
                result={"architecture":architecture,"seed":seed,"removed":removed,"case":policy,"policy":policy,"physical":False,"real_parameters":None,
                        "architecture_savings_claim":False,"evaluation":metrics,"predictions_sha256":file_sha256(folder / "predictions.npz")}
                write_json(folder / "metrics.json",result);rows.append(result)
            print(f"{key}: projected/oracle consumer ratio={errors['full_projected_over_oracle_error_ratio']:.4f}",flush=True)
    if any(file_sha256(ROOT / name)!=digest for name,digest in protected.items()): raise AssertionError("A protected source artifact changed")
    summary={"status":"completed","protocol":protocol,"rows":rows,"diagnostics":diagnostics,"selection_sha256":file_sha256(output / "selection_frozen.json"),
             "compression_optimizer_updates":0,"official_test_evaluated":False}
    write_json(output / "summary.json",summary);aggregates,effects=render(output,summary)
    print(json.dumps({"paired_effects":effects},indent=2),flush=True)
    return summary


if __name__=="__main__":
    run(read(ROOT / "configs/internal_replication.json"))
