using Pkg
Pkg.activate(".")

using Revise

##

using ProBeTune

##

i = PBTLibrary.random_fourier_modes_sampler((0., 2.), 4)


##

