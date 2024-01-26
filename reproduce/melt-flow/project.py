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


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpuq",
            help="Specify the partition to submit to."
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
def make_slab(job):
    """Run a bulk slab simulation; equilibrate in NVT"""
    from flowermd.base.system import Pack, Simulation
    from flowermd.library import LJChain, KremerGrestBeadSpring 

    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        # Create molecules and initial configuration
        chains = LJChain(
                lengths=job.doc.chain_lengths,
                num_mols=job.doc.num_chains,
        )
        system = Pack(
                molecules=chains,
                density=job.sp.number_density,
                base_units=dict(),
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
        tau_kT = job.doc.dt * job.sp.tau_kT
        job.doc.tau_kT = tau_kT
        # Set up stuff for shrinking volume step
        print("Running shrink step.")
        shrink_kT_ramp = sim.temperature_ramp(
                n_steps=job.sp.shrink_n_steps,
                kT_start=job.sp.shrink_kT,
                kT_final=job.sp.kT
        )
        sim.run_update_volume(
                final_density=job.sp.density,
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
        print("Cooling down.")
        # Cool down to final temp:
        #TODO: Double check cool-down temp.
        cooling_steps = 3e7
        cool_ramp = sim.temperature_ramp(
                n_steps=cooling_steps,
                kT_start=job.sp.kT,
                kT_final=2.4
        )
        sim.run_NVT(n_steps=cooling_steps, kT=cool_ramp, tau_kt=job.doc.tau_kT)
        sim.run_NVT(n_steps=1e7, kT=2.4, tau_kt=job.doc.tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.nvt_runs = 1
        print("Simulation finished.")


@PPS_Weld.pre(initial_nvt_run_done)
@PPS_Weld.post(nvt_equilibrated)
@PPS_Weld.operation(
        directives={"ngpu": 1, "executable": "python -u"},
        name="run-nvt-longer"
)
def run_nvt_longer(job):
    import pickle
    import unyt
    from unyt import Unit
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

        gsd_path = job.fn(f"trajectory-nvt{job.doc.nvt_runs}.gsd")
        log_path = job.fn(f"log-nvt{job.doc.nvt_runs}.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
                initial_state=job.fn("restart-nvt.gsd"),
                forcefield=ff,
                reference_values=ref_values,
                dt=job.sp.dt,
                gsd_write_freq=job.sp.gsd_write_freq,
                gsd_file_name=gsd_path,
                log_write_freq=job.sp.log_write_freq,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        sim.run_NVT(n_steps=1e7, kT=job.sp.kT, tau_kt=job.doc.tau_kT)
        sim.save_restart_gsd(job.fn("restart-nvt.gsd"))
        job.doc.nvt_runs += 1
        print("Simulation finished.")

if __name__ == "__main__":
    PPS_Weld().main()
