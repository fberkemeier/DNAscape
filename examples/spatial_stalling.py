"""Spatial pauses and permanent barriers using the existing stochastic engine."""
import numpy as np

from dnascape import rsim


def main():
    x = np.arange(256)
    initiation = np.full(x.size, .01)
    density = np.where((x >= 64) & (x < 96), .3, .01)
    recovery = .1 * (1 + .4 * np.sin(x / 20))
    # These bins cannot initiate or be crossed from either direction.
    initiation[160:168] = 0.
    density[160:168] = 1000.
    recovery[160:168] = 0.
    result = rsim(source={"initiation_rate": initiation, "fork_speed": 1.4,
                          "stall_rate_per_kb": density, "stall_recovery_rate": recovery},
                  assumptions={"sim_number": 32, "resolution_x": 1., "seed": 7,
                               "stall_shared_directions": True, "fork_speed_std": .2,
                               "time_statsQ": True, "time_grid": np.arange(301.),
                               "max_rep_time": 300., "verbose": False})
    print(f"Complete realizations: {result['completed'].sum()}/32")
    print(f"Replicated fraction at cutoff: {result['replicated_fraction'][-1]:.4f}")
    return result


if __name__ == "__main__":
    main()
