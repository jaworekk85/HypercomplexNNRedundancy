"""Post-hoc analytic control: data-weighted projection versus direct Q fit.

No new networks or inference. Full least squares satisfies Cxx A.T = Cxy.
For right-Hamilton basis B_c, the constrained design has Gram matrix
tr(Cxx) I. Its optimum is a_c = <B_c, (Cxx A.T).T>/tr(Cxx).
Thus data-weighted projection reproduces direct Q least squares. Comparing
the latter only to Euclidean/Frobenius projection is not method novelty.
"""

import json
from pathlib import Path

import numpy as np

from scripts.cifar_cnn import write_json
from scripts.internal_merging import cache_path
from src.compression.internal_merging import BASIS, expand_right, project_right
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    records = []
    for seed in summary["protocol"]["config"]["seeds"]:
        path = cache_path(seed)
        with np.load(path) as cached:
            values = cached["trained"][::2].astype(np.float64).reshape(-1,16,4)
        moments = {unit: values[:,unit].T@values[:,unit]/len(values) for unit in range(16)}
        fits_path = output / f"seed_{seed}/all_pair_fits.json"
        candidates = json.loads(fits_path.read_text(encoding="utf-8"))
        errors, unweighted = [], []
        for row in candidates:
            cxx = moments[row["keeper"]]
            full = np.asarray(row["maps"]["full_real"]["matrix"])
            direct = np.asarray(row["maps"]["quaternion_right"]["matrix"])
            denominator = np.trace(cxx)
            coefficients = np.einsum("coi,io->c",BASIS,cxx@full.T)/denominator if denominator > 0 else np.zeros(4)
            weighted = expand_right(coefficients)
            errors.append(float(np.max(np.abs(weighted-direct))))
            unweighted.append(float(np.linalg.norm(project_right(full)[1]-direct)))
        records.append({"seed":seed,"directed_pairs":len(candidates),"maximum_weighted_vs_direct_matrix_error":max(errors),
                        "median_unweighted_vs_direct_matrix_frobenius_distance":float(np.median(unweighted)),
                        "cache_sha256":file_sha256(path),"pair_fits_sha256":file_sha256(fits_path)})
    maximum = max(row["maximum_weighted_vs_direct_matrix_error"] for row in records)
    result = {"status":"passed" if maximum < 1e-10 else "failed", "post_hoc_analytic_control":True,
              "new_inference":False,"new_training":False,"records":records,"script_sha256":file_sha256(Path(__file__)),
              "interpretation":"Data-weighted projection reproduces the direct quaternion fit. Its gain over Euclidean projection cannot be presented as a new algorithm or uniquely quaternion-specific performance gain. The measured intermediate structural restriction/cost remains valid under the stated setup."}
    write_json(output / "data_weighted_projection_audit.json", result)
    print(json.dumps({"status":result["status"],"maximum_matrix_error":maximum},indent=2),flush=True)
    if result["status"]!="passed":
        raise AssertionError("Weighted projection identity failed")
    return result


if __name__=="__main__":
    run(ROOT / "results/internal_merging")
