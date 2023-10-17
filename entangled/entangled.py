import itertools

from cmeutils.gsd_utils import get_molecule_cluster
import gsd.hoomd
import hoomd
import numpy as np


def initialize_frame(gsd_file, frame_index, head_index=0, tail_index=-1):
    """"""
    with gsd.hoomd.open(gsd_file, "rb") as traj:
        frame = traj[frame_index]

    cluster, cl_props = get_molecule_cluster(snap=frame)
    n_chains=len(cluster.cluster_keys)
    types_list = [f"C{i}" for i in range(n_chains)]
    frame.particles.types = types_list

    type_ids = np.zeros_like(frame.particles.typeid)
    for idx, indices in enumerate(cluster.cluster_keys):
        type_ids[indices] = idx
    frame.particles.typeid = type_ids
    frame.particles.velocity = None # Zero out the particle velocities 

    bond_types = []
    bond_ids = []
    last_type = None
    id_count = -1

    for group in frame.bonds.group:
        type1 = frame.particles.types[frame.particles.typeid[group[0]]]
        type2 = frame.particles.types[frame.particles.typeid[group[1]]]
        bond_type = "-".join([type1, type2])
        if bond_type != last_type:
            last_type = bond_type
            bond_types.append(bond_type)
            id_count += 1
        bond_ids.append(id_count)

    frame.bonds.N = len(bond_ids)
    frame.bonds.types = bond_types
    frame.bonds.typeid = bond_ids
    # Index numbers for head and tail particles of each chain
    head_tail_indices = []
    for indices in cluster.cluster_keys:
        head_tail_indices.append(indices[head_index])
        head_tail_indices.append(indices[tail_index])
    return frame, head_tail_indices


def initialize_forcefield(
        frame,
        bond_r0=0.01,
        bond_k=100,
        bond_delta=0.0,
        pair_epsilon=1,
        pair_sigma=1,
        pair_r_cut=2.5
):
    """"""
    bond = hoomd.md.bond.FENEWCA()
    lj = hoomd.md.pair.LJ(default_r_cut=r_cut, nlist=hoomd.md.nlist.Cell(buffer=0.40))

    for btype in frame.bonds.types:
        bond.params[btype] = dict(
                k=bond_k,
                r0=bond_r0,
                delta=delta,
                epsilon=0,
                sigma=0
        )

    all_combos = list(itertools.combinations(frame.particles.types, 2))
    all_same_pairs = [(p, p) for p in frame.particles.types]
    # LJ pair interactions of inter-chain particles
    for pair in all_combos:
        lj.params[pair] = dict(epsilon=pair_epsilon, sigma=pair_sigma)
    # Same-chain particle pair interactions are turned off
    for pair in all_same_pairs:
        lj.params[pair] = dict(epsilon=0, sigma=0)
        lj.r_cut[pair] = 0
    return lj, bond


def initialize_sim(
        frame,
        forces,
        head_tail_indices,
        kT,
        tau_kT,
        dt,
        log_file_name,
        gsd_file_name,
        log_write_freq,
        gsd_write_freq
):
    """"""
    sim = hoomd.simulation.Simulation(device=hoomd.device.auto_select())
    sim.create_state_from_snapshot(frame)
    integrator = hoomd.md.Integrator(dt=dt)
    # Head and tail particles of each chain will be frozen in place
    integrate_group = hoomd.filter.SetDifference(
            hoomd.filter.All(), hoomd.filter.Tags(head_tail_indices)
    )
    method = hoomd.md.methods.NVT(kT=kT, tau=tau_kT, filter=integrate_group)
    integrator.methods.append(method)
    integrator.forces = list(forces)
    sim.operations.add(integrator)

    # Set up writers and loggers:
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    gsd_logger = hoomd.logging.Logger(categories=["scalar", "particle"])
    thermo_props = hoomd.md.compute.ThermodynamicQuantities(
            filter=hoomd.filter.All()
    )
    sim.operations.computes.append(thermo_props)
    logger.add(sim, quantities=["timestep", "tps"])
    logger.add(
            thermo_props, quantities=["potential_energy", "kinetic_energy"]
    )

    for f in forces:
        logger.add(f, quantities=["energy"])
        gsd_logger.add(f, quantities=["energy", "forces"])

    table_file = hoomd.write.Table(
            output=open(log_file_name, mode="w", newline="\n"),
            trigger=hoomd.trigger.Periodic(period=int(log_write_freq)),
            logger=logger,
            max_header_len=None
    )
    gsd_writer = hoomd.write.GSD(
            filename=gsd_file_name,
            trigger=hoomd.trigger.Periodic(int(gsd_write_freq)),
            mode="wb",
            dynamic=["momentum"],
            log=gsd_logger
    )

    sim.operations.add(gsd_writer)
    sim.operations.add(table_file)
    return sim
