from scipy.stats import qmc
import torch.distributions as dt
import numpy as np
import torch

class Cauchy(dt.cauchy.Cauchy):
    def __init__(self,generator,loc,scale,validate_args):
        super().__init__(loc,scale,validate_args)
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z

class ContinousBernoulli(dt.continuous_bernoulli.ContinuousBernoulli):
    def __init__(self,probs=None, logits=None, lims=(0.499, 0.501),generator=None, validate_args=None):
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)
        super().__init__(probs,logits,lims,validate_args)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z
    
class Exponential(dt.exponential.Exponential):
    def __init__(self,rate,generator=None, validate_args=None):
        super().__init__(rate,validate_args)
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z

    
class HalfCauchy(dt.half_cauchy.HalfCauchy):
    def __init__(self,scale, generator=None, validate_args=None):
        super().__init__(scale,validate_args)
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z

    

class HalfNormal(dt.half_normal.HalfNormal):
    def __init__(self,scale, generator=None, validate_args=None):
        super().__init__(scale,validate_args)
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z

class Laplace(dt.laplace.Laplace):
    def __init__(self,loc,scale,generator=None,validate_args=None):
        super().__init__(loc,scale,validate_args)
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z
    
class Normal(dt.normal.Normal):
    def __init__(self,loc,scale,generator=None,validate_args=None):
        super().__init__(loc,scale,validate_args)
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z
    

class Uniform(dt.uniform.Uniform):
    def __init__(self,low,high,generator=None,validate_args=None):
        super().__init__(low,high,validate_args)
        self.sampler=qmc.LatinHypercube(d=1,strength=2,seed=generator)

    def sample(self,sample_shape):
        s=np.array(sample_shape,dtype=np.int32)
        s=np.prod(s)
        s=int(s)
        u=self.sampler.random(n=s)
        u=torch.tensor(u)
        z=self.icdf(u).reshape(*sample_shape)
        return z    


    
