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
struct StateVariable
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;
  static constexpr Scalar pi = 3.141592653589793238;

  enum class Output
  {
    lowPass = 0,
    highPass,
    bandPass,
    normalizedBandPass
  };

  Scalar state[2 * avec::size<Vec>()];
  Scalar memory[avec::size<Vec>()]; // for antisaturator
  Scalar frequency[avec::size<Vec>()];
  Scalar resonance[avec::size<Vec>()];
  Scalar outputMode[avec::size<Vec>()];

  StateVariable()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    setFrequency(0.25);
    setResonance(0.0);
    reset();
  }

  void reset()
  {
    std::fill_n(state, 3 * avec::size<Vec>(), 0.0);
  }

  void setOutput(Output output, int channel)
  {
    outputMode[channel] = static_cast<int>(output);
  }

  void setOutput(Output output)
  {
    std::fill_n(outputMode, avec::size<Vec>(), static_cast<int>(output));
  }

  void setFrequency(Scalar normalized, int channel)
  {
    frequency[channel] = tan(pi * normalized);
  }

  void setFrequency(Scalar normalized)
  {
    std::fill_n(frequency, avec::size<Vec>(), tan(pi * normalized));
  }

  void setResonance(Scalar value)
  {
    std::fill_n(resonance, avec::size<Vec>(), 2.0 * (1.0 - value));
  }

  void setResonance(Scalar value, int channel)
  {
    resonance[channel] = 2.0 * (1.0 - value);
  }

  void setupNormalizedBandPass(Scalar bandwidth,
                               Scalar normalizedFrequency,
                               int channel)
  {
    auto [w, r] = normalizedBandPassPrewarp(bandwidth, normalizedFrequency);
    frequency[channel] = w;
    resonance[channel] = r;
  }

  void setupNormalizedBandPass(Scalar bandwidth, Scalar normalizedFrequency)
  {
    auto [w, r] = normalizedBandPassPrewarp(bandwidth, normalizedFrequency);
    std::fill_n(frequency, avec::size<Vec>(), w);
    std::fill_n(resonance, avec::size<Vec>(), r);
  }

  // linear

  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<-1>(input, output);
  }

  void bandPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<static_cast<int>(Output::bandPass)>(input, output);
  }

  void lowPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<static_cast<int>(Output::lowPass)>(input, output);
  }

  void normalizedBandPass(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    linear<static_cast<int>(Output::normalizedBandPass)>(input, output);
  }

  // nonlinear

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int numIterations,
                    Saturator saturate,
                    SaturationGain saturationGain,
                    SaturatorWithDerivative computeSaturationAndDerivative,
                    SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<-1>(input,
                           output,
                           numIterations,
                           saturate,
                           saturationGain,
                           computeSaturationAndDerivative,
                           saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void lowPass(VecBuffer<Vec> const& input,
               VecBuffer<Vec>& output,
               int numIterations,
               Saturator saturate,
               SaturationGain saturationGain,
               SaturatorWithDerivative computeSaturationAndDerivative,
               SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::lowPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void bandPass(VecBuffer<Vec> const& input,
                VecBuffer<Vec>& output,
                int numIterations,
                Saturator saturate,
                SaturationGain saturationGain,
                SaturatorWithDerivative computeSaturationAndDerivative,
                SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::bandPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void normalizedBandPass(
    VecBuffer<Vec> const& input,
    VecBuffer<Vec>& output,
    int numIterations,
    Saturator saturate,
    SaturationGain saturationGain,
    SaturatorWithDerivative computeSaturationAndDerivative,
    SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::normalizedBandPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

  template<class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void highPass(VecBuffer<Vec> const& input,
                VecBuffer<Vec>& output,
                int numIterations,
                Saturator saturate,
                SaturationGain saturationGain,
                SaturatorWithDerivative computeSaturationAndDerivative,
                SaturatorAutomation saturatorAutomation)
  {
    withAntisaturation<static_cast<int>(Output::highPass)>(
      input,
      output,
      numIterations,
      saturate,
      saturationGain,
      computeSaturationAndDerivative,
      saturatorAutomation);
  }

private:
  static std::pair<Scalar, Scalar> normalizedBandPassPrewarp(
    Scalar bandwidth,
    Scalar normalizedFrequency)
  {
    Scalar const b = pow(2.0, bandwidth * 0.5);
    Scalar const n0 = normalizedFrequency / b;
    Scalar const n1 = std::min(1.0, normalizedFrequency * b);
    Scalar const w0 = tan(pi * n0);
    Scalar const w1 = tan(pi * n1);
    Scalar const w = sqrt(w0 * w1);
    Scalar const r = 0.5 * w1 / w0;
    return { w, r };
  }

  template<int multimodeOutput>
  void linear(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + avec::size<Vec>());
    Vec const g = Vec().load_a(frequency);
    Vec const r = Vec().load_a(resonance);

    Vec const output_mode = Vec().load_a(outputMode);
    auto const is_high_pass = output_mode == static_cast<int>(Output::highPass);
    auto const is_band_pass = output_mode == static_cast<int>(Output::bandPass);
    auto const is_nrm_band_pass =
      output_mode == static_cast<int>(Output::normalizedBandPass);

    if constexpr (multimodeOutput == static_cast<int>(Output::highPass)) {

      for (int i = 0; i < numSamples; ++i) {

        Vec const in = input[i];

        Vec const g_r = r + g;

        Vec const high = (in - g_r * s1 - s2) / (1.0 + g_r * g);

        Vec const v1 = g * high;
        Vec const band = v1 + s1;
        s1 = band + v1;

        Vec const v2 = g * band;
        Vec const low = v2 + s2;
        s2 = low + v2;

        output[i] = high;
      }
    }

    else {

      for (int i = 0; i < numSamples; ++i) {

        Vec const in = input[i];

        Vec const band = (g * (in - s2) + s1) / (1.0 + g * (r + g));

        s1 = band + band - s1;

        Vec const v2 = g * band;
        Vec const low = v2 + s2;
        s2 = low + v2;

        if constexpr (multimodeOutput == -1) {
          Vec normalized_band = band * r;
          Vec high = in - (g * r * band + s2);

          output[i] = select(is_band_pass,
                             band,
                             select(is_nrm_band_pass,
                                    normalized_band,
                                    select(is_high_pass, high, low)));
        }
        else if constexpr (multimodeOutput ==
                           static_cast<int>(Output::lowPass)) {
          output[i] = low;
        }
        else if constexpr (multimodeOutput ==
                           static_cast<int>(Output::bandPass)) {
          output[i] = band;
        }
        else if constexpr (multimodeOutput ==
                           static_cast<int>(Output::normalizedBandPass)) {
          output[i] = band * r;
        }
        else {
          static_assert(false, "Wrong multimodeOutput.");
        }
      }
    }

    s1.store_a(state);
    s2.store_a(state + avec::size<Vec>());
  }

  template<int multimodeOutput,
           class Saturator,
           class SaturationGain,
           class SaturatorWithDerivative,
           class SaturatorAutomation>
  void withAntisaturation(
    VecBuffer<Vec> const& input,
    VecBuffer<Vec>& output,
    int numIterations,
    Saturator saturate,
    SaturationGain saturationGain,
    SaturatorWithDerivative computeSaturationAndDerivative,
    SaturatorAutomation saturatorAutomation)
  {
    int const numSamples = input.getNumSamples();
    output.setNumSamples(numSamples);

    Vec s1 = Vec().load_a(state);
    Vec s2 = Vec().load_a(state + avec::size<Vec>());
    Vec u = Vec().load_a(memory);
    Vec const g = Vec().load_a(frequency);
    Vec const r = Vec().load_a(resonance) - 2.0;

    Vec const output_mode = Vec().load_a(outputMode);
    auto const is_high_pass = output_mode == static_cast<int>(Output::highPass);
    auto const is_band_pass = output_mode == static_cast<int>(Output::bandPass);
    auto const is_nrm_band_pass =
      output_mode == static_cast<int>(Output::normalizedBandPass);

    for (int i = 0; i < numSamples; ++i) {

      saturatorAutomation();

      Vec const g_r = r + g;
      Vec const g_2 = g + g;
      Vec const d = 1.0 + g * (g_r);

      Vec const in = input[i];

      // Mystran's cheap method, solving for antisaturated bandpass "u"

      Vec const sigma = saturationGain(u); // saturate(u)/u

      u = (s1 + g * (in - s2)) / (sigma * d + g_2);

      // Newton - Raphson

      for (int it = 0; it < numIterations; ++it) {
        Vec band, delta_band_delta_u;
        computeSaturationAndDerivative(u, band, delta_band_delta_u);
        Vec const imp = band * d - g * (in - (u + u) - s2) - s1;
        Vec const delta = delta_band_delta_u * d + g_2;
        u -= imp / delta;
      }

      Vec const band = saturate(u);

      s1 = band + band - s1;

      Vec const v2 = g * band;
      Vec const low = v2 + s2;
      s2 = low + v2;

      if constexpr (multimodeOutput == -1) {
        Vec normalized_band = band * r + 2.0 * u;
        Vec high = in - (g_r * band + s2 + u);

        output[i] = select(is_band_pass,
                           band,
                           select(is_nrm_band_pass,
                                  normalized_band,
                                  select(is_high_pass, high, low)));
      }
      else if constexpr (multimodeOutput == static_cast<int>(Output::lowPass)) {
        output[i] = low;
      }
      else if constexpr (multimodeOutput ==
                         static_cast<int>(Output::bandPass)) {
        output[i] = band;
      }
      else if constexpr (multimodeOutput ==
                         static_cast<int>(Output::normalizedBandPass)) {
        output[i] = band * r + 2.0 * u;
      }
      else if constexpr (multimodeOutput ==
                         static_cast<int>(Output::highPass)) {
        output[i] = in - (g_r * band + s2 + u);
      }
      else {
        static_assert(false, "Wrong multimodeOutput.");
      }
    }

    s1.store_a(state);
    s2.store_a(state + avec::size<Vec>());
    u.store_a(memory);
  }
};

} // namespace adsp
