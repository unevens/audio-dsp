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
struct OnePole
{
  using Float = typename ScalarTypes<Vec>::Float;
  static constexpr Float pi = 3.141592653589793238;

  Float state[aavec::size<Vec><Vec>()];
  Float frequency[aavec::size<Vec><Vec>()];

  OnePole()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    setFrequency(0.25);
    reset();
  }

  void reset()
  {
    std::fill_n(state, aavec::size<Vec><Vec>(), 0.0);
  }

  void setFrequency(Float normalized, int channel)
  {
    frequency[channel] = tan(pi * normalized);
  }

  void setFrequency(Float normalized)
  {
    std::fill_n(frequency, aavec::size<Vec><Vec>(), tan(pi * normalized));
  }

  void lowPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    processBlock<1>(input, output);
  }

  void highPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    processBlock<0>(input, output);
  }

private:
  template<int isLowPass>
  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s = Vec().load_a(state);
    Vec const g = Vec().load_a(frequency);

    for (int i = 0; i < numSamples; ++i) {

      Vec const in = input[i];

      Vec const v = g * (in - s) / (1.0 + g);
      Vec const low = v + s;

      s = low + v;

      if constexpr (isLowPass) {
        output[i] = low;
      }
      else {
        output[i] = in - low;
      }
    }

    s.store_a(state);
  }
};

} // namespace adsp