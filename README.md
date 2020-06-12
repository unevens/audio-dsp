# [*adsp*](https://github.com/unevens/adsp)

*adsp* is a collection of template classes for audio dsp using SIMD instructions for x86 processors.
 
Each template class abstracts over the many SIMD instructions sets, and will choose what it's better suited for the architecture it is compiled for, given the number of audio channel you want to process in parallel and the floating point precision that you specify.

*adsp* depends on [*avec*](https://github.com/unevens/avec), a library that wraps Agner Fog's [vectorclass](https://github.com/vectorclass/version2), and provides containers and views over aligned memory.

## Splines

The file `Spline.hpp` implements cubic Hermite splines with (optionally) smoothly automatable knots.

## State Variable Filters

The file `StateVariable.hpp` implements linear and non-linear state variable filters, as described in the book [*The art of VA filter design*](https://www.discodsp.net/VAFilterDesign_2.1.2.pdf) by Vadim Zavalishin, using the cheap zero-delay method by Teemu Voipio, see https://www.kvraudio.com/forum/viewtopic.php?f=33&t=349859.

## GammaEnv

The file `GammaEnv.hpp` implements [gammaenv](https://github.com/avaneev/gammaenv), an envelope follower by Aleksey Vaneev, with some minor customizations.

## Noise Generator

The file `Noise.hpp` implements a (white) noise generator.
It uses [a SIMD implementation](https://github.com/unevens/xorshift32_16bit_simd) of a [16 bit xorshift32 random number generator](https://b2d-f9r.blogspot.com/2010/08/16-bit-xorshift-rng-now-with-more.html
), to generate 4 samples of noise in parallel. 

It is the only part of *adsp* which is not header only, as it needs the file `xorshift32_16bit_simd.cpp` to be compiled.

## OnePole 

The file `OnePole.hpp` implements simple one pole filters, also as described in *The art of VA filter design*.

## SimpleHighPass

The file `SimpleHighPass.hpp` implements naive one pole high pass filter, suitable for DC removal.


## Biquad Filters

The file `Biquad.hpp` implements simple biquad filters from the RBJ cookbook.
