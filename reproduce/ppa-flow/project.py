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


class PPS_Weld(FlowProject):
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


@PPS_Weld.label
def initial_nvt_run_done(job):
    return job.doc.nvt_runs >= 1


@PPS_Weld.label
def nvt_equilibrated(job):
    return job.doc.nvt_equilibrated


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


def density_adjustment(job):
    refs = get_ref_values(job)
    void_d = job.doc.void_particle_size
    void_d_real = void_d * refs["length"]
    void_d_cm = void_d_real.to("cm")
    void_r_cm = void_d_cm / 2
    density = job.doc.mass_g / (4/3*np.pi*(void_r_cm)**3)
    return density

@PPS_Weld.post(initial_nvt_run_done)
@PPS_Weld.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="make-slab"
)
def make_slab(job):
    """Run a bulk slab simulation; equilibrate in NVT"""
    from flowermd.base.system import Pack
    from flowermd.library import PPS, OPLS_AA_PPS
    from flowermd.modules.welding import SlabSimulation
    from flowermd.modules.welding import add_void_particles

    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        # Create molecules and initial configuration
        pps = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)
        system = Pack(
                molecules=pps,
                density=job.sp.density,
                base_units=dict(),
        )
        system.apply_forcefield(
                force_field=OPLS_AA_PPS(),
                auto_scale=True,
                r_cut=2.5,
                scale_charges=True,
                remove_hydrogens=job.sp.remove_hydrogens,
                remove_charges=job.sp.remove_charges,
                pppm_resolution=(64, 64, 64),
                pppm_order=4,
        )
        # Store reference units and values
        job.doc.total_mass_amu = system.mass
        job.doc.ref_mass = system.reference_mass.to("amu").value
        job.doc.ref_mass_unit = "amu"
        job.doc.ref_energy = system.reference_energy.to("kJ/mol").value
        job.doc.ref_energy_unit = "kJ/mol"
        job.doc.ref_length = (
                system.reference_length.to("nm").value * job.sp.sigma_scale
        )
        job.doc.ref_length_unit = "nm"
        # Set dt
        if job.sp.remove_hydrogens:
            job.doc.dt = 0.0003
        else:
            job.doc.dt = 0.0001

        gsd_path = job.fn(f"trajectory{job.doc.nvt_runs + 1}.gsd")
        log_path = job.fn(f"log{job.doc.nvt_runs + 1}.txt")
        init_snap = system.hoomd_snapshot
        hoomd_ff = system.hoomd_forcefield
        job.doc.N_particles = init_snap.particles.N
        # Add void particles if needed
        # Find correct value of void particle sigma
        if job.sp.void_size:
            Lx = system.target_box[0]
            job.doc.interface_area = Lx**2
            void_r = (Lx**2 / np.pi)**(1/2) # A = pi(r^2)
            void_diameter = (void_r * 2) * job.sp.void_size
            job.doc.void_particle_size = void_diameter * job.sp.sigma_scale
            init_snap, hoomd_ff = add_void_particles(
                    snapshot=init_snap,
                    forcefield=hoomd_ff,
                    num_voids=1,
                    void_axis=(1,0,0),
                    void_diameter=void_diameter,
                    epsilon=0.25,
                    r_cut=void_diameter
            )
        # Set up initial simulation
        sim = SlabSimulation(
                initial_state=init_snap,
                forcefield=hoomd_ff,
                dt=job.doc.dt,
                wall_sigma=1.0,
                wall_epsilon=0.5,
                wall_r_cut=1.12,
                r_cut=2.5,
                reference_values=get_ref_values(job),
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
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"
        job.doc.mass_g = sim.mass.to("g")
        #density_adj = density_adjustment(job)
        #job.doc.density_adj = density_adj

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
