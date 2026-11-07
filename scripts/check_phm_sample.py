"""Independent tiny CPU review of the pinned public PHM sample, not its training.

Only the previously inspected ComplexPHMLinear class is extracted from the
notebook AST. Notebook setup, downloads and training cells are not executed.
"""
import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6982112ef33f0c81cc81aab900bfc63340be6565"


def run():
    path = ROOT / ".cache/literature/PHM_Sample_Code.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in notebook["cells"] if "class ComplexPHMLinear" in "".join(c.get("source", [])))
    tree = ast.parse(cell)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ComplexPHMLinear")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_PHM_class_only", "exec"), namespace)
    cls = namespace["ComplexPHMLinear"]
    rng = np.random.default_rng(2026)
    rows = []
    for bases in (2, 3):
        n, m = 3, 4
        matrices = np.stack([np.eye(2), np.array([[0., -1.], [1., 0.]]), np.diag([1., -1.])])[:bases]
        factors = rng.integers(-8, 9, size=(bases, n, m)).astype(float)
        weight = sum(np.kron(h, a) for h, a in zip(matrices, factors))
        # Independent block-ordered orthogonal projection onto the declared basis.
        blocks = weight.reshape(2, n, 2, m).transpose(0, 2, 1, 3).reshape(4, n*m)
        design = matrices.reshape(bases, 4).T
        optimal = np.linalg.lstsq(design, blocks, rcond=None)[0].reshape(bases, n, m)
        correct_weight = sum(np.kron(h, a) for h, a in zip(matrices, optimal))
        model = cls(2*m, 2*n, B=bases, bias=False).double()
        model.project_from_dense(torch.tensor(weight, dtype=torch.float64))
        projected = sum(torch.kron(model.H[b], model.As[b]) for b in range(bases))
        probe = torch.tensor(rng.integers(-4, 5, size=(5, 2*m)), dtype=torch.float64)
        with torch.no_grad():
            actual = model(probe)
            declared = probe@projected.T
            expected = probe@torch.tensor(weight, dtype=torch.float64).T
        torch.testing.assert_close(actual, declared, atol=1e-12, rtol=1e-12)
        independent_error = float(np.linalg.norm(correct_weight-weight))
        sample_error = float(torch.linalg.norm(projected-torch.tensor(weight)))
        if independent_error > 1e-10 or sample_error < 1e-3:
            raise AssertionError("Counterexample no longer matches inspected sample")
        rows.append({"bases": bases, "weight_shape": list(weight.shape), "exactly_representable_input": True,
                     "correct_block_projection_frobenius_error": independent_error,
                     "sample_projection_frobenius_error": sample_error,
                     "sample_forward_mse_vs_target": float((actual-expected).square().mean()),
                     "actual_forward_matches_declared_block_layout": True,
                     "matmul_flops_ratio_sample_vs_dense": bases/2,
                     "flops_scope": "Matmul count for two half-size GEMMs per basis in this forward, excluding extra elementwise work; not measured latency"})
    result = {"status": "counterexample_reproduced", "commit": COMMIT,
              "source_url": f"https://github.com/milab-nsu/PHM/blob/{COMMIT}/Sample_Code.ipynb",
              "cached_notebook_sha256": file_sha256(path), "cases": rows,
              "conclusion": "Sample projection uses an interleaved partition inconsistent with its blocked forward. This is a counterexample for this pinned sample, not a reproduction or invalidation of published model results. The sample forward alone does not imply FLOP savings from reduced parameter count."}
    write_json(ROOT / ".cache/literature/phm_sample_review.json", result)
    print(json.dumps({"status": result["status"], "cases": rows}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    run()
