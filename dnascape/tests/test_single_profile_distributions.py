"""Named profiles work in distribution views without duplicate sample inputs."""
import json

import numpy as np
import pytest

from dnascape.workflows.preview import PreviewLine, build_preview_plot_options


@pytest.mark.parametrize("kind,density", [("Histogram",False),("Histogram",True),("Boxplot",False)])
@pytest.mark.parametrize("log_x", [False,True])
@pytest.mark.parametrize("log_y", [False,True])
def test_single_named_distribution_matches_explicit_samples(kind,density,log_x,log_y):
    samples=np.array([1.,2.,3.,4.,5.,6.,7.,8.,np.nan])
    options=dict(plot_type=kind,title="",x_label="Value",y_label="Count",color="#2596be",
        palette="viridis",histogram_bins=4,density=density,show_legend=True,
        show_colorbar=False,log_y=log_y,invert_y=False,y_limits=None,series_label="Profile",
        log_x=log_x)
    explicit=build_preview_plot_options(values=samples,**options)
    named=build_preview_plot_options(lines=(PreviewLine("Profile",np.arange(len(samples)),samples,"#2596be"),),**options)
    assert named['series'][0]['data']==explicit['series'][0]['data']
    assert named['xAxis']==explicit['xAxis']
    assert named['yAxis']==explicit['yAxis']
    assert named['series'][0]['name']=='Profile'
    json.dumps(named,allow_nan=False)
