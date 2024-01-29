#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.

"""

import signac
import flow
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    ''''''
    parameters = OrderedDict()
    # System and model params:
    # MN is the combo of number of chains (M) and chain lengths (N)
    parameters["MN"] = [
            #(50, 50),
            #(500, 50),
            #(500, 100),
            #(250, 200),
            #(200, 350),
            (500, 500),
            #(200, 700),
    ]
    parameters["number_density"] = [0.85] # 1/sigma^3
    parameters["epsilon"] = [1.0]
    parameters["sigma"] = [1.0]
    parameters["bond_k"] = [30]
    parameters["bond_max"] = [1.5] # units of sigma
    parameters["bond_delta"] = [0]
    # Run time params:
    parameters["dt"] = [0.0012]
    parameters["friction_coeff"] = [4.2]
    parameters["kT"] = [4.2]
    parameters["n_steps"] = [1e8]
    parameters["shrink_kT"] = [6.0]
    parameters["shrink_n_steps"] = [1e6]
    parameters["shrink_period"] = [1]
    parameters["tau_kT"] = [100]
    parameters["gsd_write_freq"] = [1e5]
    parameters["log_write_freq"] = [1e4]
    parameters["sim_seed"] = [42]
    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project() # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        statepoint = dict(zip(param_names, params))
        job = project.open_job(statepoint)
        job.init()
        job.doc.setdefault("num_chains", job.sp.MN[0])
        job.doc.setdefault("chain_lengths", job.sp.MN[1])
        job.doc.setdefault("equilibrated", False)
        job.doc.setdefault("runs", 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
