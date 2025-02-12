# ABCMethods

A Julia package featuring various ABC methods.

## Methods

This package implements

- Classic ABC: See the historical paper [Pritchard et al. - 1999](https://academic.oup.com/mbe/article/16/12/1791/2925409)

- ABC-SMC: A sequential Monte Carlo version of the classic ABC method [Moral et al. - 2012](https://link.springer.com/article/10.1007/s11222-011-9271-y). Note that this is the only method not using a reference table for training.

- ABC-CNN: ABC method as described by [Åkesson et al. - 2021](https://ieeexplore.ieee.org/abstract/document/9525290).

- ABC-Conformal: ABC method completely free of summary statistics and threshold selection as described in [Baragatti et al. - 2024](https://arxiv.org/abs/2406.04874). It use Approximate Bayesian Computation (ABC) with deep learning and conformal prediction.

## Example of usage

The documentation is not (yet) available however, detailed Quarto notebook on several examples using all implemented ABC methods are available.

- [MA(2) example](https://mistea.pages.mia.inra.fr/codes_articles/abcdconformal/julia/MA2/ABC_Conformal_MA2.html) (2 parameters) -> Well known toy Bayesian example.
- [Discrete Lotka-Volterra example](https://mistea.pages.mia.inra.fr/codes_articles/abcdconformal/julia/LoktaVolterra/ABC_Conformal_LV.html) (3 parameters) -> Very challenging for some extreme parameters.
- [Phytoplankton dynamics in Lake with a toy model example](https://mistea.pages.mia.inra.fr/codes_articles/abcdconformal/julia/Lake_norm/ABC_Conformal_Lake.html) (9 parameters) -> High dimensional example.


## Installation

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