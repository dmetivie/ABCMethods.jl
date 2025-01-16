# ABCMethods

A Julia package featuring various ABC methods. 

The package is currently NOT in the Julia general registry (will be soon), but on a local registry, hence to download it just add:

```julia
using Pkg
pkg"registry add https://github.com/dmetivie/LocalRegistry"
```

and then `add` it as a normal package

```julia
Pkg.add("ABCMethods")
# or
pkg> add ABCMethods
```

This package implements 

- Classic ABC: See the historical paper [Pritchard et al. - 1999](https://academic.oup.com/mbe/article/16/12/1791/2925409)

- ABC-SMC: A sequential Monte Carlo version of the classic ABC method [Moral et al. - 2012](https://link.springer.com/article/10.1007/s11222-011-9271-y). Note that this is the only method not using a reference table for training.

- ABC-CNN: ABC method as described by [Åkesson et al. - 2021](https://ieeexplore.ieee.org/abstract/document/9525290).

- ABC-Conformal: ABC method completely free of summary statistics and threshold selection as described in [Baragatti et al. - 2024](https://arxiv.org/abs/2406.04874). It use Approximate Bayesian Computation (ABC) with deep learning and conformal prediction.

Example of applications can be found in Quarto notebooks attached with the paper. In particular for the MA(2) example [see here](https://forgemia.inra.fr/mistea/codes_articles/abcdconformal/-/tree/main/julia/MA2) (we should have a better display soon).