# Learned PHM basis compression pilot

Status: completed.

## Hypothesis

After physical channel removal, readapting learned PHM consumer bases may improve held-out output reconstruction and repaired classification beyond preserving the bases. Reinitialization probes dependence on the learned starting span. This hypothesis and the procedure are not established novelty or a demonstrated gain.

## Measured results

| Case | Stored reals | Trainable reals | Validation before (%) | Validation after (%) | CE after |
|---|---:|---:|---:|---:|---:|
| compressed_preserve | 233418 | 232266 | 86.70 | 90.20 | 0.30305 |
| compressed_adapt | 233418 | 232394 | 87.80 | 90.50 | 0.30382 |
| compressed_reinitialize | 233418 | 232394 | 87.90 | 90.00 | 0.304644 |
| uncompressed_preserve | 272746 | 271594 | 91.50 | 91.60 | 0.27219 |
| uncompressed_adapt | 272746 | 271722 | 91.50 | 91.60 | 0.272155 |
| compact_scratch | 233418 | 233418 | 9.90 | 90.40 | 0.294454 |

| Block | Policy | Initial check MSE | Fitted check MSE | Best fit step |
|---|---|---:|---:|---:|
| 0 | preserve | 1.27976 | 0.610153 | 300 |
| 0 | adapt | 1.27976 | 0.580998 | 300 |
| 0 | reinitialize | 5.40775 | 0.585606 | 300 |
| 8 | preserve | 0.589743 | 0.28207 | 299 |
| 8 | adapt | 0.589743 | 0.276958 | 300 |
| 8 | reinitialize | 0.887724 | 0.276958 | 300 |

## Interpretation and limits

One CE-selected PHM parent, seed2026, two independent local fitting sites (blocks0/8), fixed n=rank=4, one 50% internal-group removal fraction. The combined compressed model changes only these two hidden widths, not all nine sites; reported full-model savings must reflect this. Six local fits, three compressed repairs, two uncompressed further-training controls and one full-budget compact scratch reference; no automatic extension.

All ordinary surviving weights, BN affine parameters and PHM cores can train. All PHM bases are frozen except the TWO chosen conv2 bases in adapt/reinitialize. Preserve stores the same coefficients but trains128fewer reals. Both uncompressed controls start from the unchanged selected parent and follow the same preserve/adapt flags,3epochs and paired streams; no local fitting on the uncompressed controls. BN statistics update under the same recipe. Epoch3 is the final repair endpoint; no checkpoint or hyperparameter selection on check/validation.

One full200epoch scratch model has the exact compact physical hidden widths but randomly initializes all ordinary parameters and learns all18PHM bases using the original baseline recipe/seed. It is a full-training-budget reference, not an equal-update repair control; report CE-selected and final validation separately. Same train/validation split, no official test. It does not inherit the parent weights, basis tensors or pruned channel identities beyond chosen widths.

48000 training images,1000 validation,500 fit,500 check; verify disjointness and exact coverage. No official-test predictions. Record before/after validation CE/accuracy, held-out local MSE, actual stored/trainable reals, buffers, model bytes, optimizer attempts/successes, factor changes and effective-kernel changes. Raw basis-coordinate distance is not an invariant scientific measure because PHM factorization has change-of-basis freedom. Local error does not guarantee whole-network ordering.

Primary comparison is adapt versus preserve after identical3epoch repair. Continue-to-replication only if mean per-block held-out relative MSE reduction is>=2%, neither block worsens by>1%, validation accuracy improves>=0.5pp, compressed CE decreases, and the compressed CE improvement is at least the uncompressed further-training CE improvement. Also report a local-only-signal flag independently. Thresholds are practical pilot gates, not significance tests. Retain every failed/negative outcome. Scratch comparison provides budget/context, not a claimed matched-update effect. No automatic next dataset/seeds/allocation run.

Gate outcome: `{"status": "evaluated", "continue_to_replication": false, "checks": {"local_heldout_signal": true, "accuracy_gain": false, "lower_compressed_ce": false, "compression_specific_ce_gain": false}, "relative_check_mse_gains": [0.04778274378796458, 0.018125275883948307], "mean_relative_check_mse_gain": 0.03295400983595644, "validation_accuracy_gain": 0.0030000000000000027, "compressed_ce_gain": -0.0007700023651123211, "uncompressed_ce_gain": 3.564071655276502e-05, "compression_specific_ce_gain": -0.0008056430816650861, "statistical_significance_claimed": false, "automatic_followup": false}`.

Scratch row uses the CE-selected checkpoint; its final validation is separately retained in result.json.
One parent seed, two local sites, one split; no significance, general superiority or novelty claim.
Prior art: He2016ResNet, He2017ChannelPruning, Kim2020, VieiraValle2022ELM, Singh2026, ZhangEtAl2021, Grassucci2024, IbnAhad2026HypercomplexQuantization, Liebenwein2021ALDS, Frantar2022SPDY (references.bib).
