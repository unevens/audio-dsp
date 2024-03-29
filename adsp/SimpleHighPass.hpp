/*
Copyright 2020-2021 Dario Mambro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once
#include "avec/Avec.hpp"

namespace adsp {

template<class Vec>
struct SimpleHighPass final
{
  using Float = typename ScalarTypes<Vec>::Float;

  Float inputMemory[avec::size<Vec>()];
  Float outputMemory[avec::size<Vec>()];
  Float alpha[avec::size<Vec>()];

  void setHighPassFrequency(Float frequency)
  {
    std::fill_n(alpha, avec::size<Vec>(), exp(-frequency));
    if (frequency == 0.0) {
      reset();
    }
  }

  void setHighPassFrequency(Float frequency, int channel)
  {
    alpha[channel] = exp(-frequency);
    if (frequency == 0.0) {
      inputMemory[channel] = outputMemory[channel] = 0.0;
    }
  }

  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    Vec in_mem = Vec().load_a(inputMemory);
    Vec out_mem = Vec().load_a(outputMemory);
    Vec a = Vec().load_a(alpha);

    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    for (int i = 0; i < numSamples; ++i) {
      Vec const io = input[i];
      out_mem = a * (out_mem + io - in_mem);
      in_mem = io;
      output[i] = out_mem;
    }

    in_mem.store_a(inputMemory);
    out_mem.store_a(outputMemory);
  }

  void reset() { std::fill_n(inputMemory, 2 * avec::size<Vec>(), 0.0); }

  SimpleHighPass()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    reset();
  }
};

} // namespace adsp