"""Exercise the added architecture, nested surgery, and equivalent control."""

from unittest.mock import patch

import numpy as np
import pytest

torch=pytest.importorskip("torch")
from scripts import internal_replication as replication
from src.compression.internal_merging import fit_pairs,select_pairs,substitute_features
from src.compression.weighted_projection import add_weighted_maps,compact_policy
from src.evaluation.block_ablation import capture_blocks,grouped_spatial,predict_from_block


def test_fourblock_nested_compaction_control_and_checkpoint(tmp_path):
    torch.manual_seed(72)
    config={"quaternion_widths":[2,4,5,6],"dropout":[.1,.2,.3,.3]}
    source={"architecture":"cnn8","config":config}
    model=replication.make_model(source).eval()
    images=torch.randn(9,3,32,32)
    features=capture_blocks(model,images,9)[1]
    values=grouped_spatial(features,True)
    candidates=fit_pairs(values[:5],model.blocks[2][0].kernel.detach().double().numpy())
    pairs=add_weighted_maps(select_pairs(candidates,2),values[:5])
    for removed in (1,2):
        selected=pairs[:removed]
        direct,kept,_=compact_policy(model,selected,"quaternion_exact")
        weighted,weighted_kept,_=compact_policy(model,selected,"weighted_projected")
        assert kept==weighted_kept and len(kept)==4-removed
        expected=predict_from_block(model,1,substitute_features(features,selected,"quaternion_right"))
        with torch.no_grad():
            actual=direct(images).numpy()
            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=1e-5)
            np.testing.assert_allclose(weighted(images).numpy(),actual,atol=2e-6,rtol=1e-5)
        path=tmp_path / f"removed_{removed}.pt"
        torch.save({"model":direct.state_dict()},path)
        manifest={"source":source,"kept_units":kept,"checkpoint_path":path.name}
        with patch.object(replication,"ROOT",tmp_path),torch.no_grad():
            restored=replication.load_case(manifest,torch.device("cpu"))
            assert torch.equal(restored(images),direct(images))
            assert len(restored.blocks)==4 and restored.blocks[2][0].in_channels==4-removed
    production=replication.make_model({"architecture":"cnn8","config":{"quaternion_widths":[8,16,24,32],"dropout":[.1,.2,.3,.3]}})
    assert sum(p.numel() for p in production.parameters())==118058
    assert sum(b.numel() for b in production.buffers())==1288
