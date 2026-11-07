import numpy as np
import pytest

torch=pytest.importorskip("torch")
from src.compression.internal_merging import fit_linear,project_right
from src.compression.weighted_projection import moment_projection


def test_data_weighted_projection_is_the_strong_equivalent_control():
    rng=np.random.default_rng(26)
    x=rng.normal(size=(300,4))@np.diag([.2,1.,3.,7.])+[1.,2.,.5,4.]
    y=x@rng.normal(size=(4,4))+rng.normal(size=(300,4))*.1
    full=np.asarray(fit_linear(x,y,"full_real")["matrix"])
    direct=np.asarray(fit_linear(x,y,"quaternion_right")["matrix"])
    weighted=moment_projection(full,x.T@x/len(x))
    np.testing.assert_allclose(weighted,direct,atol=1e-12)
    assert np.linalg.norm(project_right(full)[1]-direct)>.1
