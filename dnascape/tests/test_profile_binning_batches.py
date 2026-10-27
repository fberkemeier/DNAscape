"""Batch binning agrees with independent interval-overlap calculations."""
import numpy as np
import pytest

from dnascape.workflows.profile_grid import rebin_spatial_profile


def reference(values, intervals, starts, ends, aggregation):
    output=[]
    widths=np.diff(intervals,axis=1).reshape(-1)
    for start,end in zip(starts,ends):
        overlap=np.maximum(0.,np.minimum(intervals[:,1],end)-np.maximum(intervals[:,0],start))
        row=[]
        for column in values.T:
            valid=np.isfinite(column)&(overlap>0)
            if not np.any(valid):row.append(np.nan);continue
            signal=column[valid]
            if aggregation=='mean':value=np.sum(signal*overlap[valid])/np.sum(overlap[valid])
            elif aggregation=='sum':value=np.sum(signal*overlap[valid]/widths[valid])
            elif aggregation=='min':value=np.min(signal)
            else:value=np.max(signal)
            row.append(value)
        output.append(row)
    return np.asarray(output)


@pytest.mark.parametrize('aggregation',['mean','sum','min','max'])
@pytest.mark.parametrize('step',[.125,.75,1.,2.,3.75,200.])
@pytest.mark.parametrize('gaps',[False,True])
def test_fractional_bins_gaps_missing_values_and_partial_regions(aggregation,step,gaps):
    rng=np.random.default_rng(42)
    widths=rng.uniform(.25,2.,60)
    starts=np.r_[0.,np.cumsum(widths[:-1])]
    if gaps:starts+=np.arange(len(starts))*.35
    intervals=np.column_stack([starts,starts+widths])
    values=rng.normal(size=(60,3));values[::5,0]=np.nan;values[:,1]=np.nan;values[::7,2]=np.inf
    original=values.copy()
    result,positions,bins=rebin_spatial_profile(values,starts,intervals,start=125.,end=(intervals[-1,1]-.125)*1000,
        bin_size=step*1000,aggregation=aggregation,space_time=True)
    np.testing.assert_allclose(result,reference(values,intervals,bins[:,0],bins[:,1],aggregation),rtol=2e-12,atol=2e-12,equal_nan=True)
    np.testing.assert_equal(values,original)
    np.testing.assert_equal(positions,bins[:,0])


@pytest.mark.parametrize('aggregation',['mean','sum','min','max'])
def test_large_aligned_bins_cross_batch_boundaries(aggregation):
    values=np.arange(140_002.,dtype=float)
    positions=np.arange(len(values),dtype=float)
    intervals=np.column_stack([positions,positions+1])
    result,_,_=rebin_spatial_profile(values,positions,intervals,bin_size=2000,aggregation=aggregation)
    groups=values.reshape(-1,2)
    expected=getattr(np,aggregation)(groups,axis=1)
    np.testing.assert_array_equal(result,expected)


def test_finer_bins_and_small_signals_after_large_values():
    values=np.array([1e20,1.,2.,3.]);positions=np.arange(4.)
    intervals=np.column_stack([positions,positions+1])
    result,_,_=rebin_spatial_profile(values,positions,intervals,bin_size=2000)
    np.testing.assert_array_equal(result,[5e19,2.5])
    values=np.arange(80.);positions=np.arange(80.)
    intervals=np.column_stack([positions,positions+1])
    result,_,_=rebin_spatial_profile(values,positions,intervals,bin_size=1)
    np.testing.assert_allclose(result,np.repeat(values,1000),rtol=1e-12,atol=1e-12)


def test_binning_checks_cancellation_between_batches():
    values=np.arange(100_000.);positions=values.copy();intervals=np.column_stack([positions,positions+1])
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls==3:raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError,match='cancelled'):
        rebin_spatial_profile(values,positions,intervals,bin_size=500,cancel=cancel)
