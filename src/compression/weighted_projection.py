"""Data-moment projection control for zero-offset right-Hamilton fits."""

from copy import deepcopy

import numpy as np

from src.compression.internal_merging import BASIS,expand_right,merge_internal


def moment_projection(full_matrix,second_moment):
    full=np.asarray(full_matrix,dtype=np.float64)
    moment=np.asarray(second_moment,dtype=np.float64)
    if full.shape!=(4,4) or moment.shape!=(4,4) or not np.isfinite(full).all() or not np.isfinite(moment).all():
        raise ValueError("Expected finite full map and keeper second moment")
    denominator=float(np.trace(moment))
    coefficients=np.einsum("coi,io->c",BASIS,moment@full.T)/denominator if denominator>0 else np.zeros(4)
    return expand_right(coefficients)


def add_weighted_maps(pairs,fit_values):
    result=deepcopy(pairs)
    flat=np.asarray(fit_values,dtype=np.float64).reshape(-1,fit_values.shape[-2],4)
    for row in result:
        x=flat[:,row["keeper"]]
        moment=x.T@x/len(x)
        matrix=moment_projection(row["maps"]["full_real"]["matrix"],moment)
        row["maps"]["weighted_projected"]={"matrix":matrix.tolist(),"intercept":False,
            "maximum_difference_from_direct":float(np.max(np.abs(matrix-np.asarray(row["maps"]["quaternion_right"]["matrix"]))))}
    return result


def weighted_pairs(pairs):
    result=deepcopy(pairs)
    for row in result:
        row["maps"]["quaternion_right"]=deepcopy(row["maps"]["weighted_projected"])
    return result


def compact_policy(model,pairs,policy):
    if policy=="weighted_projected":
        return merge_internal(model,1,weighted_pairs(pairs),"quaternion_exact")
    return merge_internal(model,1,pairs,policy)
