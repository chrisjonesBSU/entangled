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
def initial_ppa_run_done(job):
    return job.doc.ppa_runs >= 1


@KG_PPA.label
def equilibrated(job):
    return job.doc.equilibrated


@KG_PPA.label
def system_ready(job):
    return job.isfile("init.gsd")


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


@KG_PPA.post(system_ready)
@KG_PPA.operation(
        directives={"ngpu": 0, "executable": "python -u"}, name="make-system"
)
def make_system(job):
    """Run a bulk slab simulation; equilibrate in NVT"""
    import unyt as u
    from flowermd.base import Pack, Lattice
    from flowermd.library import LJChain

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
        #system = Pack(
        #        molecules=chains,
        #        density=density,
        #        base_units=dict(),
        #        edge=5.0,
        #        overlap=5.0,
        #        packing_expand_factor=15
        #)
        n = int((job.sp.MN[0]/2)**(0.5))
        job.doc.lattice_n = n
        system = Lattice(
                molecules=chains,
                base_units=dict(),
                x=4,
                y=4,
                n=n,
                z_length_padding=4
        )
        job.doc.N_particles = system.n_particles
        system.to_gsd(file_name=job.fn("init.gsd"))
        print("init.gsd has been saved...")


@KG_PPA.pre(system_ready)
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
        density = job.sp.number_density * u.Unit("nm**-3")
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
        hoomd_ff = forcefield.hoomd_forces
        # Set up initial simulation
        sim = Simulation(
                initial_state=job.fn("init.gsd"),
                forcefield=hoomd_ff,
                dt=job.sp.dt / 100,
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
                n_steps=3e6, kT=job.sp.shrink_kT, tau_kt=job.doc.tau_kT
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
        sim.dt = job.sp.dt
        print("Shrinking finished.")
        print("Running Langevin simulation.")
        sim.run_langevin(
                n_steps=job.sp.n_steps,
                kT=job.sp.kT,
                default_gamma=job.sp.friction_coeff,
                default_gamma_r=(
                    job.sp.friction_coeff,
                    job.sp.friction_coeff,
                    job.sp.friction_coeff,
                )
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


@KG_PPA.pre(equilibrated)
@KG_PPA.post(initial_ppa_run_done)
@KG_PPA.operation(
        directives={"ngpu": 1, "executable": "python -u"},
        name="run-ppa"
)
def run_ppa(job):
    import pickle
    import flowermd
    from flowermd.base.simulation import Simulation
    import gsd.hoomd
    import entangled as ent
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        # Modify the starting snapshot
        ppa_snap, head_tail_indices = ent.initialize_frame(
                gsd_file=job.fn("restart.gsd"), frame_index=0
        )
        ppa_lj, ppa_bond = ent.initialize_forcefield(
                frame=ppa_snap,
                bond_r0=1.2,
                bond_k=100,
                pair_epsilon=job.sp.epsilon,
                pair_sigma=job.sp.sigma,
                pair_r_cut=1.12
        )
        gsd_path = job.fn(f"ppa_trajectory{job.doc.ppa_runs + 1}.gsd")
        log_path = job.fn(f"ppa_log{job.doc.ppa_runs + 1}.txt")
        # Create flowerMD simulation obj:
        sim = Simulation(
                initial_state=ppa_snap,
                forcefield=[ppa_lj, ppa_bond],
                reference_values=dict(),
                dt=job.sp.dt / 2, # Start with smaller dt
                gsd_write_freq=job.sp.gsd_write_freq,
                gsd_file_name=gsd_path,
                log_write_freq=job.sp.log_write_freq,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        # Set integrate group to not integrate head and tail particles
        integrate_group = hoomd.filter.SetDifference(
                hoomd.filter.All(), hoomd.filter.Tags(head_tail_indices)
        )
        sim.integrate_group = integrate_group
        sim.save_restart_gsd(job.fn("ppa_restart.gsd"))
        sim.pickle_forcefield(job.fn("ppa_forcefield.pickle"))
        # Initial relaxation run with high friction coeff
        sim.run_langevin(
                n_steps=1000,
                kT=0.001,
                default_gamma=20,
                default_gamma_r=(20, 20, 20)
        )
        # Longer run to reach equilibration
        sim.run_langevin(
                n_steps=5e5,
                kT=0.001,
                default_gamma=job.sp.friction_coeff,
                default_gamma_r=(
                    job.sp.friction_coeff,
                    job.sp.friction_coeff,
                    job.sp.friction_coeff,
                )
        )
        sim.save_restart_gsd(job.fn("ppa_restart.gsd"))
        job.doc.ppa_runs += 1
        print("PPA simulation finished.")

if __name__ == "__main__":
    KG_PPA(environment=Fry).main()
