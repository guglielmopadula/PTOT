if __name__ == "__main__":
    from ptot.distributions import Normal
    from numpy.random import Generator,PCG64
    from scipy.stats import qmc
    rng = Generator(PCG64())
    sampler=qmc.LatinHypercube(d=1,strength=2,seed=rng)
    a=Normal(rng,0,1).sample([5,5])
