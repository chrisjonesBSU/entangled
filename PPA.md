# Forcefield
FENE Spring:
k = 100epsilon / sigma^2
intra-chain pair interactions are disabled
inter-chain pair interactions are retained

# Simulation
dt = 0.006tau
kT = 0.001
1. Run for 1,000 steps:
	- High value of Gamma (20) -- same for gamma_r

2. Next 5x10^5 steps:
	- Reduce Gamma back to the standard value of 0.5


# Steps
1. Start with an equilibrated melt configuration.
2. Set the velocity for all particles to zero.
3. 
