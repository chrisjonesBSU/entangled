"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os
import unyt
from unyt import Unit
import numpy as np


class KG_PPA(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpu",
            help="Specify the partition to submit to."
        )
        parser.add_argument(
            "--exclude",
            default="gpu105",
            help="Specify the type of nodes to exclude."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )


@KG_PPA.label
def initial_run_done(job):
    return job.doc.runs >= 1


@KG_PPA.label
def equilibrated(job):
    return job.doc.equilibrated


def get_ref_values(job):
    ref_length = job.doc.ref_length * Unit(job.doc.ref_length_unit)
    ref_mass = job.doc.ref_mass * Unit(job.doc.ref_mass_unit)
    ref_energy = job.doc.ref_energy * Unit(job.doc.ref_energy_unit)
    ref_values_dict = {
            "length": ref_length,
            "mass": ref_mass,
            "energy": ref_energy
    }
    return ref_values_dict


@KG_PPA.post(initial_run_done)
@KG_PPA.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="first-run"
)
def initial_run(job):
    """Run a bulk slab simulation; equilibrate in NVT"""
    import unyt as u
    from flowermd.base import Pack, Simulation
    from flowermd.library import LJChain, KremerGrestBeadSpring
    from flowermd.utils import get_target_box_number_density

    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        # Create molecules and initial configuration
        chains = LJChain(
                lengths=job.doc.chain_lengths,
                num_mols=job.doc.num_chains,
                bond_lengths={"A-A": 0.90}
        )
        density = job.sp.number_density * u.Unit("nm**-3")
        system = Pack(
                molecules=chains,
                density=density,
                base_units=dict(),
                edge=5.0,
                overlap=5.0,
                packing_expand_factor=8
        )
        forcefield = KremerGrestBeadSpring(
                bond_k=job.sp.bond_k,
                bond_max=job.sp.bond_max,
                radial_shift=job.sp.bond_delta,
                sigma=job.sp.sigma,
                epsilon=job.sp.epsilon,
                bead_name="A"
        )

        gsd_path = job.fn(f"trajectory{job.doc.runs + 1}.gsd")
        log_path = job.fn(f"log{job.doc.runs + 1}.txt")
        init_snap = system.hoomd_snapshot
        hoomd_ff = forcefield.hoomd_forces
        job.doc.N_particles = init_snap.particles.N
        # Set up initial simulation
        sim = Simulation(
                initial_state=init_snap,
                forcefield=hoomd_ff,
                dt=job.sp.dt,
                gsd_write_freq=job.sp.gsd_write_freq,
                log_write_freq=job.sp.log_write_freq,
                gsd_file_name=gsd_path,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        sim.pickle_forcefield(job.fn("forcefield.pickle"))
        # Store more unit information in job doc
        job.doc.tau_kT = job.sp.dt * job.sp.tau_kT
        # Set up stuff for shrinking volume step
        print("Running shrink step.")
        sim.run_NVT(
                n_steps=2e6, kT=job.sp.shrink_kT, tau_kt=job.doc.tau_kT
        )
        target_box = get_target_box_number_density(
                density=density,
                n_beads=job.doc.N_particles
        )
        shrink_kT_ramp = sim.temperature_ramp(
                n_steps=job.sp.shrink_n_steps,
                kT_start=job.sp.shrink_kT,
                kT_final=job.sp.kT
        )
        sim.run_update_volume(
                final_box_lengths=target_box,
                n_steps=job.sp.shrink_n_steps,
                period=job.sp.shrink_period,
                tau_kt=job.doc.tau_kT,
                kT=shrink_kT_ramp
        )
        print("Shrinking finished.")
        print("Running NVT simulation.")
        sim.run_NVT(
                n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=job.doc.tau_kT
        )
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.runs = 1
        print("Simulation finished.")


@KG_PPA.pre(initial_run_done)
@KG_PPA.post(equilibrated)
@KG_PPA.operation(
        directives={"ngpu": 1, "executable": "python -u"},
        name="run-longer"
)
def run_longer(job):
    import pickle
    import flowermd
    from flowermd.base.simulation import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting NVT simulation...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            ff = pickle.load(f)

        gsd_path = job.fn(f"trajectory{job.doc.runs + 1}.gsd")
        log_path = job.fn(f"log{job.doc.runs + 1}.txt")
        sim = Simulation(
                initial_state=job.fn("restart.gsd"),
                forcefield=ff,
                reference_values=dict(),
                dt=job.sp.dt,
                gsd_write_freq=job.sp.gsd_write_freq,
                gsd_file_name=gsd_path,
                log_write_freq=job.sp.log_write_freq,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        sim.run_NVT(n_steps=1e7, kT=job.sp.kT, tau_kt=job.doc.tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.runs += 1
        print("Simulation finished.")

if __name__ == "__main__":
    KG_PPA().main()
