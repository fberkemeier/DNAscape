# DNAscape Mapping: edit the controls below, then run this file.
# Extract the runnable bundle first; it contains the inputs and DNAscape source.
# Run: python mapping.py   (after installing requirements.txt)
# Or:  uv run mapping.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     'numpy==2.5.3',
#     'pandas==3.0.6',
#     'scipy==1.18.1',
#     'matplotlib==3.11.2',
# ]
# ///

from pathlib import Path
from math import inf

from dnascape.workflows.mapping_recipe import run_mapping_recipe
from dnascape.workflows.recipe_exports import save_mapping_outputs, plot_mapping_outputs

HERE = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = HERE / "outputs"

# Mapping controls. Positions/resolution are in kb; times are in minutes.
# Files are relative to this script. Edit paths, selected columns/array keys,
# input forms, constants, regions, solver settings and enabled overrides here.
MAPPING = {'method': 'timing_and_speed_to_firing_rate',
 'target_shape': None,
 'settings': {'resolution': 1.0,
              'perQ': False,
              'fit_step': 2.0,
              'maxiter': 10,
              'err_threshold': 15.0,
              'refine_edge_kb': 1000.0,
              'sim_number': 128,
              'stall_rate_per_kb': 0.0,
              'neighbourhood_range': 2000.0},
 'setting_overrides': {'error': {'enabled': False, 'value': 0.0},
                       'deviation_threshold': {'enabled': False, 'value': 0.0},
                       'space_fit_step': {'enabled': False, 'value': 1e-12},
                       'space_maxiter': {'enabled': False, 'value': 1},
                       'tau': {'enabled': False, 'value': 1e-12}},
 'region': {'start_kb': 0.0,
            'end_kb': 1000.0,
            'whole_genome': False,
            'missing_policy': 'error'},
 'combination_mode': 'pairwise',
 'inputs': {'replication_timing': {'enabled': True,
                                   'form': 'x',
                                   'mode': 'upload',
                                   'files': [{'path': 'inputs/01-01-rdata_rt_wgencodeuwrepliseqk562wavesignalrep1 '
                                                      '(30).tsv',
                                              'name': 'rdata_rt_wgencodeuwrepliseqk562wavesignalrep1 '
                                                      '(30).tsv',
                                              'column': 'chr1',
                                              'array_key': None,
                                              'source_format': 'tsv'}],
                                   'track': {'chromosome': '',
                                             'bin_size': 1000,
                                             'start': None,
                                             'end': None,
                                             'aggregation': 'mean',
                                             'value_column': '',
                                             'missing': 'nan',
                                             'fill_value': 0.0,
                                             'interpolation': 'none',
                                             'smooth_window': 1,
                                             'normalize': 'none'}},
            'fork_speed': {'enabled': True,
                           'form': 'scalar',
                           'mode': 'constant',
                           'values': 1.4}}}

# Raw data controls: output, filename, format, compression and batch overrides.
# Use output="primary" to select the route's primary output. Formats:
# TSV, TXT, CSV, NPY, NPV, NPZ. Compression: none, gzip, zip, 7z, xz.
EXPORT_RAW = True
RAW_EXPORT = {'output': 'firing_rate',
 'filename': 'rmap_fr',
 'file_format': 'TSV',
 'compress': False,
 'compression': 'gzip',
 'name_style': 'short',
 'per_dataset': {}}

# Plotting is optional. The HTML preserves the app's ECharts plot settings.
# It works without NiceGUI. Change output/view, axes, colours and layout here.
PLOT = False
PLOTTING = {'output': 'firing_rate',
 'layout': 'Combined',
 'view': 'Default',
 'name_style': 'short',
 'settings': {'normalization': 'None',
              'title': 'Firing rate',
              'series_label': 'Firing rate',
              'x_label': 'Position (kb)',
              'y_label': 'Firing rate',
              'color': '#2c7f77',
              'secondary_color': '#b03412',
              'palette': 'viridis',
              'histogram_bins': 60,
              'line_width': 1.8,
              'show_markers': False,
              'show_colorbar': True,
              'log_y': False,
              'log_z': False,
              'invert_y': False,
              'log_x': False,
              'invert_x': False,
              'invert_z': False,
              'density': False,
              'show_legend': True,
              'legend_position': 'Top right',
              'show_histogram_statistics': True,
              'histogram_statistics_position': 'Top right',
              'histogram_bin_size': None,
              'histogram_x_limits': None,
              'histogram_touching': True,
              'histogram_boundary': 'left',
              'show_grid': True,
              'y_limits': None},
 'colors': [],
 'plot_export_dpi': 150}

def main():
    results = run_mapping_recipe(MAPPING, base_path=HERE)
    if EXPORT_RAW:
        for path in save_mapping_outputs(results, RAW_EXPORT, directory=OUTPUT_DIRECTORY):
            print(f"Saved {path}")
    if PLOT:
        for path in plot_mapping_outputs(results, PLOTTING, directory=OUTPUT_DIRECTORY,
                                         echarts_path=HERE / "echarts.min.js"):
            print(f"Saved {path}")

if __name__ == "__main__":
    main()
