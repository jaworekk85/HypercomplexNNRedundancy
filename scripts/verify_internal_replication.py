"""Check CNN8 training provenance and every compact replication artifact."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.internal_merging import consumer_errors
from scripts.internal_replication import load_case,make_model,read
from scripts.verify_internal_merging import independently_folded_addition,mse_from_moments
from src.cnn_training import evaluate,prepare_batch,seed_everything
from src.compression.internal_merging import grouped_to_tensor,substitute_features
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.evaluation.block_ablation import capture_blocks,classification_metrics,predict_from_block
from src.models.quaternion_conv_torch import hamilton_kernel

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def verify(output):
    output=Path(output)
    summary,frozen=read(output / "summary.json"),read(output / "selection_frozen.json")
    protocol=summary["protocol"];config=protocol["config"];checks={}

    def check(name,value):
        checks[name]=bool(value)

    check("completed_and_no_compression_updates_or_test",summary["status"]=="completed" and summary["compression_optimizer_updates"]==0 and not summary["official_test_evaluated"])
    check("original_cell_marked_exploratory",protocol["original_cell_is_not_independent_replication"])
    check("expected_case_counts",len(frozen["manifests"])==180 and len(summary["rows"])==204 and len(summary["diagnostics"])==12)
    check("selection_frozen",frozen["all_choices_frozen_before_compressed_validation"] and file_sha256(output / "selection_frozen.json")==summary["selection_sha256"])
    check("protocol_preserved",read(output / "study_protocol.json")==protocol)
    for name,digest in {**protocol["source_hashes"],**protocol["protected_inputs"]}.items():
        check(f"unchanged:{name}",file_sha256(ROOT / name)==digest)
    training=read(output / "architecture_b/training_protocol.json")
    check("training_protocol_matches_study",training["study"]==config and not training["official_test_evaluated"])
    for name,digest in training["source_hashes"].items():
        check(f"training_source:{name}",file_sha256(ROOT / name)==digest)
    check("training_data_source_preserved",file_sha256(ROOT / config["data_source"] / "summary.json")==training["source_summary_sha256"])
    check("training_split_preserved",file_sha256(ROOT / config["data_source"] / "split_indices.npz")==training["split_sha256"])
    check("training_statistics_preserved",file_sha256(ROOT / config["data_source"] / "data.json")==training["statistics_sha256"])
    check("expected_architecture_seed_sources",{(row["architecture"],row["seed"]) for row in protocol["sources"]}=={(arch,seed) for arch in ("cnn6","cnn8") for seed in config["seeds"]})
    basis=np.stack([hamilton_kernel(torch.tensor(row,dtype=torch.float64).reshape(4,1,1,1,1)).reshape(4,4).numpy() for row in np.eye(4)])
    with np.load(ROOT / config["data_source"] / "split_indices.npz") as saved:
        reference_ids={name:saved[name] for name in ("train","calibration","validation")}
    sources={}
    for source in protocol["sources"]:
        arch,seed=source["architecture"],source["seed"];tag=f"{arch}:{seed}"
        folder=ROOT / source["path"]
        checkpoint=torch.load(folder / "quaternion/model.pt",map_location="cpu",weights_only=True)
        state=checkpoint["model"]
        with np.load(folder / "split_indices.npz") as saved:
            ids={name:saved[name] for name in reference_ids}
        check(f"{tag}:identical_disjoint_splits",all(np.array_equal(ids[name],value) for name,value in reference_ids.items()) and len(np.unique(np.concatenate(list(ids.values()))))==50000)
        check(f"{tag}:identical_normalization",file_sha256(folder / "data.json")==training["statistics_sha256"])
        if arch=="cnn8":
            metrics=read(folder / "quaternion/metrics.json")
            last=torch.load(folder / "quaternion/last.pt",map_location="cpu",weights_only=True)
            record=read(folder / "source.json")
            check(f"{tag}:training_sources_and_config",last["config"]==checkpoint["config"]==source["config"] and last["source_hashes"]==metrics["source_hashes"]==training["source_hashes"])
            check(f"{tag}:source_record",record==source and file_sha256(folder / "quaternion/model.pt")==source["checkpoint_sha256"] and file_sha256(ROOT / source["cache_path"])==source["cache_sha256"])
            history=last["history"]
            check(f"{tag}:fifty_complete_epochs",last["epoch"]==50 and [row["epoch"] for row in history]==list(range(1,51)))
            expected_attempts=50*((len(ids["train"])+source["config"]["batch_size"]-1)//source["config"]["batch_size"])
            check(f"{tag}:optimizer_attempts",last["updates"]==metrics["optimizer_updates_attempted"]==expected_attempts==18750)
            check(f"{tag}:full_resume_state",all(name in last for name in ("optimizer","scheduler","scaler","order_rng","augment_rng","torch_rng","cuda_rng")) and last["scheduler"]["last_epoch"]==50 and len(last["cuda_rng"])>0)
            steps=[float(item["step"]) for item in last["optimizer"]["state"].values()]
            check(f"{tag}:tracked_optimizer_states",len(steps)==len(metrics["first_update"]["parameters"]) and min(steps)>0 and max(steps)<=expected_attempts)
            audit=last["audit"]
            check(f"{tag}:actual_first_update",audit==metrics["first_update"] and all(audit["checks"].values()) and len(audit["quaternion_layers"])==8 and all(row["max_abs_update"]>0 and row["gradient_norm"]>0 for row in audit["parameters"]) and all(value>0 for item in audit["quaternion_layers"].values() for values in item.values() for value in values))
            chosen=next((row for row in history if row["epoch"]==last["best_epoch"]),None)
            check(f"{tag}:minimum_validation_ce_selection",chosen is not None and checkpoint["best_epoch"]==metrics["best_epoch"]==last["best_epoch"] and abs(min(row["validation_loss"] for row in history)-last["best_loss"])<1e-12 and abs(chosen["validation_loss"]-metrics["selected_validation"]["cross_entropy"])<1e-12)
            check(f"{tag}:best_state_exported",set(state)==set(last["best_state"]) and all(torch.equal(value,last["best_state"][name]) for name,value in state.items()))
            with (folder / "quaternion/training.csv").open(encoding="utf-8",newline="") as handle:
                csv_rows=list(csv.DictReader(handle))
            check(f"{tag}:csv_matches_checkpoint",len(csv_rows)==50 and all(all(float(saved[name])==value for name,value in row.items()) for saved,row in zip(csv_rows,history)))
            check(f"{tag}:no_official_test_inference",not metrics["test_evaluated"] and not source["official_test_evaluated"] and not (folder / "quaternion/test_predictions.npz").exists())
            check(f"{tag}:baseline_budget",metrics["real_trainable_parameters"]==118058 and metrics["nontrainable_buffer_elements"]==1288 and metrics["checkpoint_reload_max_error"]<=1e-6)
            del last
        with np.load(ROOT / source["cache_path"]) as saved:
            values=saved["trained"].astype(np.float64)
        check(f"{tag}:spatial_cache_shape",values.shape==(1000,64,16,4) and np.isfinite(values).all())
        fit_values=values[::2].reshape(-1,64)
        moment=fit_values.T@fit_values/len(fit_values)
        candidates=read(ROOT / config["source_initial_study"] / f"seed_{seed}/all_pair_fits.json") if arch=="cnn6" else read(output / f"conditions/{arch}/seed_{seed}/all_pair_fits.json")
        check(f"{tag}:all_directed_pairs",{(row["keeper"],row["victim"]) for row in candidates}=={(a,b) for a in range(16) for b in range(16) if a!=b})
        max_mse,max_normal,max_projection=0.,0.,0.
        for pair in candidates:
            for mode,item in pair["maps"].items():
                matrix=np.asarray(item["matrix"])
                mse,normal=mse_from_moments(moment,pair["keeper"],pair["victim"],matrix)
                max_mse=max(max_mse,abs(mse-item["fit_feature_mse"]))
                if mode=="quaternion_right":
                    coeff=np.einsum("oi,coi->c",matrix,basis)/4
                    max_projection=max(max_projection,float(np.max(np.abs(matrix-np.einsum("c,coi->oi",coeff,basis)))))
                    normal=np.einsum("coi,io->c",basis,normal)
                max_normal=max(max_normal,float(np.max(np.abs(normal))))
                _,discarded,energy=independently_folded_addition(state["blocks.2.0.kernel"],pair["victim"],matrix,basis)
                max_projection=max(max_projection,abs(discarded-item["projection"]["discarded_squared_norm"]),abs(energy-item["projection"]["full_squared_norm"]))
            outgoing_energy=float(state["blocks.2.0.kernel"][:,:,pair["victim"]].double().square().sum())
            max_mse=max(max_mse,abs(pair["fit_selection_score"]-pair["maps"]["full_real"]["fit_feature_mse"]*outgoing_energy))
        check(f"{tag}:fit_mse_and_score",max_mse<1e-8)
        check(f"{tag}:normal_equations",max_normal<1e-8)
        check(f"{tag}:family_and_projection",max_projection<1e-8)
        ordered=[];used=set()
        for pair in sorted(candidates,key=lambda row:(row["fit_selection_score"],row["keeper"],row["victim"])):
            if pair["keeper"] not in used and pair["victim"] not in used:
                ordered.append(pair);used.update((pair["keeper"],pair["victim"]))
        check(f"{tag}:eight_disjoint_pairs_available",len(ordered)==8 and len(used)==16)
        for removed in config["removed_units"]:
            key=f"{tag}:{removed}";pairs=frozen["pairs"][key]
            check(f"{key}:fit_only_nested_selection",len(pairs)==removed and [(r["keeper"],r["victim"]) for r in pairs]==[(r["keeper"],r["victim"]) for r in ordered[:removed]])
            errors=[]
            for pair,original in zip(pairs,ordered):
                check(f"{key}:original_maps:{pair['keeper']}:{pair['victim']}",all(pair[name]==original[name] for name in ("keeper","victim","fit_selection_score")) and all(pair["maps"][mode]==original["maps"][mode] for mode in ("quaternion_right","full_real")))
                x=values[::2,:,pair["keeper"]].reshape(-1,4)
                xx=x.T@x/len(x);full=np.asarray(pair["maps"]["full_real"]["matrix"])
                coefficients=np.array([np.trace(b.T@full@xx) for b in basis])/np.trace(xx)
                weighted=np.einsum("c,coi->oi",coefficients,basis)
                errors.extend((float(np.max(np.abs(weighted-np.asarray(pair["maps"]["weighted_projected"]["matrix"])))),float(np.max(np.abs(weighted-np.asarray(pair["maps"]["quaternion_right"]["matrix"]))))))
            check(f"{key}:independent_weighted_control",max(errors)<=config["weighted_projection_tolerance"])
        if arch=="cnn6":
            old_pairs=read(ROOT / config["source_initial_study"] / "selection_frozen.json")["pairs_by_seed"][str(seed)]
            pairs=frozen["pairs"][f"{tag}:4"]
            check(f"{tag}:exploratory_cell_reused",[(p["keeper"],p["victim"]) for p in pairs]==[(p["keeper"],p["victim"]) for p in old_pairs])
        sources[(arch,seed)]={"source":source,"state":state,"ids":ids,"values":values}

    torch.set_num_threads(config["cpu_threads"]);seed_everything(config["seeds"][0]);device=torch.device(config["device"])
    (images,labels),unused_test=load_cached_cifar(ROOT / config["archive"]);del unused_test
    stats=read(ROOT / config["data_source"] / "data.json")["statistics"]
    mean=torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    std=torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    lookup={(r["architecture"],r["seed"],r["removed"],r["case"]):r for r in frozen["manifests"]}
    for (arch,seed),source in sources.items():
        tag=f"{arch}:{seed}";state=source["state"];ids=source["ids"]
        original=make_model(source["source"]).to(device).eval();original.load_state_dict(state)
        split_ids={"check_calibration":ids["calibration"][1::2],"validation":ids["validation"]}
        check_features=grouped_to_tensor(source["values"][1::2].astype(np.float32))
        val_features=capture_blocks(original,prepare_batch(torch.as_tensor(images[ids["validation"]],device=device),mean,std),config["batch_size"])[1]
        actual=capture_blocks(original,prepare_batch(torch.as_tensor(images[split_ids["check_calibration"][:32]],device=device),mean,std),32)[1]
        check(f"{tag}:cache_matches_selected_checkpoint",float((actual-check_features[:32]).abs().max())<=config["equivalence_tolerance"])
        for removed in config["removed_units"]:
            key=f"{tag}:{removed}";pairs=frozen["pairs"][key];models={}
            case_base=output / f"conditions/{arch}/seed_{seed}/removed_{removed:02d}"
            rows=[row for row in summary["rows"] if (row["architecture"],row["seed"],row["removed"])==(arch,seed,removed)]
            check(f"{key}:all_prespecified_cases",len(rows)==17 and sum(r["policy"]=="random" for r in rows)==10)
            for row in rows:
                folder=case_base / row["case"];name=f"{key}:{row['case']}"
                check(f"{name}:metrics_record",read(folder / "metrics.json")==row)
                check(f"{name}:predictions_hash",file_sha256(folder / "predictions.npz")==row["predictions_sha256"])
                if row["physical"]:
                    manifest=lookup[(arch,seed,removed,row["case"])]
                    check(f"{name}:frozen_manifest",read(folder / "manifest.json")==manifest and file_sha256(folder / "manifest.json")==row["manifest_sha256"] and manifest["pairs"]==pairs)
                    check(f"{name}:exploratory_label",manifest["reused_exploratory_condition"]==(arch=="cnn6" and removed==4))
                    path=ROOT / manifest["checkpoint_path"]
                    check(f"{name}:checkpoint_hash_and_bytes",file_sha256(path)==manifest["checkpoint_sha256"] and path.stat().st_size==row["checkpoint_bytes"])
                    kept=manifest["kept_units"];channels=[component*16+unit for component in range(4) for unit in kept]
                    current=torch.load(path,map_location="cpu",weights_only=True)["model"]
                    changed={"blocks.1.3.kernel","blocks.1.4.weight","blocks.1.4.bias","blocks.1.4.running_mean","blocks.1.4.running_var","blocks.2.0.kernel"}
                    check(f"{name}:unchanged_other_state",set(current)==set(state) and all(torch.equal(value,current[k]) for k,value in state.items() if k not in changed))
                    slices=torch.equal(current["blocks.1.3.kernel"],state["blocks.1.3.kernel"][:,kept])
                    slices &= all(torch.equal(current[f"blocks.1.4.{suffix}"],state[f"blocks.1.4.{suffix}"][channels]) for suffix in ("weight","bias","running_mean","running_var"))
                    check(f"{name}:physical_producer_and_bn_slicing",slices)
                    expected=state["blocks.2.0.kernel"][:,:,kept].double().numpy().copy()
                    if row["policy"] in ("quaternion_exact","full_projected","weighted_projected"):
                        mode={"quaternion_exact":"quaternion_right","full_projected":"full_real","weighted_projected":"weighted_projected"}[row["policy"]]
                        for pair in pairs:
                            addition,_,_=independently_folded_addition(state["blocks.2.0.kernel"],pair["victim"],np.asarray(pair["maps"][mode]["matrix"]),basis)
                            expected[:,:,kept.index(pair["keeper"])]+=addition
                    check(f"{name}:independent_consumer_folding",np.allclose(current["blocks.2.0.kernel"].numpy(),expected,atol=1e-6,rtol=1e-5))
                    if row["policy"]=="random":
                        victims=np.random.default_rng(config["random_seed_offset"]+seed*100+manifest["random_repetition"]).permutation(16)[:removed]
                        check(f"{name}:prespecified_nested_random",kept==sorted(set(range(16))-set(victims)))
                    else:
                        expected_kept=list(range(16)) if row["policy"]=="baseline" else sorted(set(range(16))-{p["victim"] for p in pairs})
                        check(f"{name}:prespecified_kept_units",kept==expected_kept)
                    model=load_case(manifest,device)
                    base_params,base_buffers,per_unit=(73898,902,1736) if arch=="cnn6" else (118058,1288,1448)
                    n=0 if row["policy"]=="baseline" else removed
                    check(f"{name}:actual_parameter_and_buffer_budget",sum(p.numel() for p in model.parameters())==row["real_parameters"]==base_params-n*per_unit and sum(b.numel() for b in model.buffers())==row["buffer_elements"]==base_buffers-8*n)
                    check(f"{name}:compact_dimensions",model.blocks[1][3].out_channels==model.blocks[2][0].in_channels==16-n and len(model.blocks)==(3 if arch=="cnn6" else 4))
                    check(f"{name}:no_masks_or_extra_map_parameters",row["masked_parameters"]==row["additional_inference_map_parameters"]==0)
                    check(f"{name}:graph_probe",manifest["probe_error"]<=config["equivalence_tolerance"])
                    if row["policy"] in ("quaternion_exact","full_projected","paired_pruned"):
                        models[row["policy"]]=model
                else:
                    check(f"{name}:functional_oracle_accounting",row["real_parameters"] is None and not row["architecture_savings_claim"])
                with np.load(folder / "predictions.npz") as saved:
                    for split,image_ids in split_ids.items():
                        check(f"{name}:{split}:ids_and_labels",np.array_equal(saved[f"{split}_ids"],image_ids) and np.array_equal(saved[f"{split}_labels"],labels[image_ids]))
                        if row["physical"]:
                            metrics,logits=evaluate(model,torch.as_tensor(images[image_ids],device=device),torch.as_tensor(labels[image_ids],device=device),mean,std,config["batch_size"])
                        else:
                            features=check_features if split=="check_calibration" else val_features
                            mode="quaternion_right" if row["policy"]=="quaternion_substitution" else "full_real"
                            logits=predict_from_block(original,1,substitute_features(features.to(device),pairs,mode),batch_size=config["batch_size"])
                            metrics=classification_metrics(logits,labels[image_ids])
                        check(f"{name}:{split}:reload_predictions",np.array_equal(logits,saved[f"{split}_logits"]))
                        check(f"{name}:{split}:recomputed_metrics",all(abs(metrics[k]-row["evaluation"][split][k])<=1e-10 for k in metrics))
            diagnostic=next(r for r in summary["diagnostics"] if (r["architecture"],r["seed"],r["removed"])==(arch,seed,removed))
            check(f"{key}:diagnostic_record",read(case_base / "consumer_diagnostics.json")==diagnostic)
            errors=consumer_errors(original,models,check_features,pairs,config["batch_size"])
            check(f"{key}:consumer_diagnostics_reproduced",all(abs(value-diagnostic["consumer_check"][k])<=1e-10 for k,value in errors.items()))
            check(f"{key}:direct_q_consumer_ratio",abs(errors["quaternion_error"]/errors["full_oracle_error"]-diagnostic["quaternion_over_oracle_error_ratio"])<1e-12)
            check(f"{key}:exact_q_composition",errors["quaternion_closure_max_abs_error"]<=config["equivalence_tolerance"])
            discarded,total=0.,0.
            for pair in pairs:
                _,a,b=independently_folded_addition(state["blocks.2.0.kernel"],pair["victim"],np.asarray(pair["maps"]["full_real"]["matrix"]),basis)
                discarded+=a;total+=b
            check(f"{key}:added_energy_projection",abs(discarded/total-diagnostic["discarded_full_energy_fraction"])<1e-10)
            for split in split_ids:
                with np.load(case_base / "weighted_projected/predictions.npz") as weighted,np.load(case_base / "quaternion_exact/predictions.npz") as direct,np.load(case_base / "quaternion_substitution/predictions.npz") as oracle:
                    err=float(np.max(np.abs(weighted[f"{split}_logits"]-direct[f"{split}_logits"])))
                    check(f"{key}:{split}:equivalent_strong_control",err<=config["equivalence_tolerance"] and abs(err-diagnostic[f"{split}_weighted_logit_error"])<1e-10)
                    check(f"{key}:{split}:functional_vs_physical_q",np.max(np.abs(oracle[f"{split}_logits"]-direct[f"{split}_logits"]))<=config["equivalence_tolerance"])
            print(f"Verified all artifacts for {key}",flush=True)

    aggregates=read(output / "aggregated.json")
    check("expected_aggregates",len(aggregates)==28)
    for aggregate in aggregates:
        values=[];ces=[];prefix=f"aggregate:{aggregate['architecture']}:{aggregate['removed']}:{aggregate['policy']}"
        for seed in config["seeds"]:
            rows=[r for r in summary["rows"] if (r["architecture"],r["removed"],r["seed"],r["policy"])==(aggregate["architecture"],aggregate["removed"],seed,aggregate["policy"])]
            values.append(float(np.mean([r["evaluation"]["validation"]["accuracy"] for r in rows])))
            ces.append(float(np.mean([r["evaluation"]["validation"]["cross_entropy"] for r in rows])))
        check(prefix,abs(np.mean(values)-aggregate["mean_accuracy"])<1e-12 and abs(np.std(values,ddof=1)-aggregate["sample_sd_accuracy"])<1e-12 and abs(np.mean(ces)-aggregate["mean_cross_entropy"])<1e-12 and all(row["seed"]==seed and row["accuracy"]==value and row["cross_entropy"]==ce for row,seed,value,ce in zip(aggregate["seed_values"],config["seeds"],values,ces)))
        values=[];ces=[]
        for seed in config["seeds"]:
            rows=[r for r in summary["rows"] if (r["architecture"],r["removed"],r["seed"],r["policy"])==(aggregate["architecture"],aggregate["removed"],seed,aggregate["policy"])]
            values.append(float(np.mean([r["evaluation"]["check_calibration"]["accuracy"] for r in rows])))
            ces.append(float(np.mean([r["evaluation"]["check_calibration"]["cross_entropy"] for r in rows])))
        check(prefix+":secondary_check",abs(np.mean(values)-aggregate["check_mean_accuracy"])<1e-12 and abs(np.std(values,ddof=1)-aggregate["check_sample_sd_accuracy"])<1e-12 and abs(np.mean(ces)-aggregate["check_mean_cross_entropy"])<1e-12 and all(row["seed"]==seed and row["accuracy"]==value and row["cross_entropy"]==ce for row,seed,value,ce in zip(aggregate["check_seed_values"],config["seeds"],values,ces)))
    for effect in read(output / "paired_effects.json"):
        left,right=[next(r for r in aggregates if (r["architecture"],r["removed"],r["policy"])==(effect["architecture"],effect["removed"],policy)) for policy in (effect["left"],effect["right"])]
        values=[100*(a["accuracy"]-b["accuracy"]) for a,b in zip(left["seed_values"],right["seed_values"])]
        ces=[a["cross_entropy"]-b["cross_entropy"] for a,b in zip(left["seed_values"],right["seed_values"])]
        check(f"effect:{effect['architecture']}:{effect['removed']}:{effect['left']}:{effect['right']}",np.allclose(values,effect["seed_differences_pp"],atol=1e-12,rtol=0) and abs(np.mean(values)-effect["mean_pp"])<1e-12 and abs(np.std(values,ddof=1)-effect["sample_sd_pp"])<1e-12)
        check(f"ce_effect:{effect['architecture']}:{effect['removed']}:{effect['left']}:{effect['right']}",np.allclose(ces,effect["seed_ce_differences"],atol=1e-12,rtol=0) and abs(np.mean(ces)-effect["mean_ce_difference"])<1e-12)
    result={"status":"passed" if all(checks.values()) else "failed","checks":checks,"check_count":len(checks),"verifier_sha256":file_sha256(Path(__file__)),"official_test_evaluated":False}
    write_json(output / "verification.json",result)
    print(json.dumps({"status":result["status"],"checks":len(checks),"failed":[k for k,v in checks.items() if not v]},indent=2),flush=True)
    if result["status"]!="passed":
        raise AssertionError("Replication artifact verification failed; retain every failed check")
    return result


if __name__=="__main__":
    verify(ROOT / "results/internal_replication")
