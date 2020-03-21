/**
 * Simd implementation of Aleksey Vaneev's https://github.com/avaneev/gammaenv
 *
 * Differences from the original:
 * - does peak/rms computation and can compute the output in dB
 * - IsIverse is not supported
 * - Attack and Release are not in seconds but in angular frequency units
 * (2*pi/(samplerate*seconds))
 *
 * By Dario Mambro @ https://github.com/unevens/avec
 */

/**
 * Copyright (c) 2016 Aleksey Vaneev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "avec/Avec.hpp"
#include <math.h>

namespace adsp {

/**
 * "gammaenv" produces smoothed-out S-curve envelope signal with the specified
 * attack and release characteristics. The attack and release times can be
 * further adjusted in real-time. Delay parameter is also specified as the
 * percentage of the total time.
 *
 * The S-curve produced by this envelope algorithm closely resembles a
 * sine-wave signal slightly augmented via the tanh() function. Such
 * augmentation makes the shape slightly steeper and in the end allows the
 * algorithm to follow it closer. The name "gammaenv" relates to this
 * algorithm's version.
 *
 * The algorithm's topology is based on 5 sets of "leaky integrators" (the
 * simplest form of 1st order low-pass filters). Each set (except the 5th) use
 * 4 low-pass filters in series. Outputs of all sets are then simply
 * summed/subtracted to produce the final result. The topology is numerically
 * stable for any valid input signal, but may produce envelope overshoots
 * depending on the input signal.
 *
 * Up to 25% of total attack (or release) time can be allocated (via Delay
 * parameters) to the delay stage. The delay is implemented by the same
 * topology: this has the benefit of not requiring additional memory
 * buffering. For example, it is possible to get up to 250 ms delay on
 * a 1-second envelope release stage without buffering.
 *
 * The processSymm() function provides the "perfect" implementation of the
 * algorithm, but it is limited to the same attack and release times. A more
 * universal process() function can work with any attack and release times,
 * but it is about 2 times less efficient and the actual attack stage's
 * envelope can range from the "designed" U to the undesired sharp V shape.
 * Unfortunately, the author was unable to find an approach that could be
 * similar to the processSymm() function while providing differing attack and
 * release times (the best approach found so far lengthens the delay stage
 * unpredictably).
 */

template<class Vec>
struct GammaEnv final
{
  using Scalar = typename ScalarTypes<Vec>::Scalar;
  using Mask = typename MaskTypes<Vec>::Mask;

  Scalar env[16 * Vec::size()];
  Scalar enva[4 * Vec::size()];
  Scalar envb[4 * Vec::size()];
  Scalar envr[16 * Vec::size()];
  Scalar env5[Vec::size()];
  Scalar enva5[Vec::size()];
  Scalar envb5[Vec::size()];
  Scalar envr5[Vec::size()];
  Scalar prevr[Vec::size()];

  Scalar useRms[Vec::size()];

  struct VecData final
  {
    Vec env[16];
    Vec enva[4];
    Vec envb[4];
    Vec envr[16];
    Vec env5;
    Vec enva5;
    Vec envb5;
    Vec envr5;
    Vec prevr;
    Mask rms;
    Vec to_db_coef;

    VecData(GammaEnv const& gammaEnv)
      : env5(Vec().load_a(gammaEnv.env5))
      , enva5(Vec().load_a(gammaEnv.enva5))
      , envb5(Vec().load_a(gammaEnv.envb5))
      , envr5(Vec().load_a(gammaEnv.envr5))
      , prevr(Vec().load_a(gammaEnv.prevr))
      , rms(Vec().load_a(gammaEnv.useRms) != 0.0)
    {
      to_db_coef = select(
        rms, 10.0 / 2.30258509299404568402, 20.0 / 2.30258509299404568402);

      for (int i = 0; i < 4; ++i) {
        enva[i] = Vec().load_a(gammaEnv.enva + i * Vec::size());
        envb[i] = Vec().load_a(gammaEnv.envb + i * Vec::size());
      }
      for (int i = 0; i < 16; ++i) {
        env[i] = Vec().load_a(gammaEnv.env + i * Vec::size());
        envr[i] = Vec().load_a(gammaEnv.envr + i * Vec::size());
      }
    }

    void update(GammaEnv& gammaEnv) const
    {
      for (int i = 0; i < 4; ++i) {
        enva[i].store_a(gammaEnv.enva + i * Vec::size());
        envb[i].store_a(gammaEnv.envb + i * Vec::size());
      }
      for (int i = 0; i < 16; ++i) {
        env[i].store_a(gammaEnv.env + i * Vec::size());
        envr[i].store_a(gammaEnv.envr + i * Vec::size());
      }
      env5.store_a(gammaEnv.env5);
      enva5.store_a(gammaEnv.enva5);
      envb5.store_a(gammaEnv.envb5);
      envr5.store_a(gammaEnv.envr5);
      prevr.store_a(gammaEnv.prevr);
    }

    Vec process(Vec in) { return process_<0>(in); }

    Vec processDB(Vec in) { return process_<1>(in); }

    template<int outputInDB = 0>
    Vec process_(Vec in)
    {
      in = select(rms, in * in, abs(in));
      env[0] += (in - env[0]) * enva[0];
      env[1] += (env5 - env[1]) * enva[1];
      env[2] += (env[4 * 3 + 1] - env[2]) * enva[2];
      env[3] += (env[4 * 3 + 0] - env[3]) * enva[3];
      env5 += (env[4 * 3 + 0] - env5) * enva5;
      int i;

      for (i = 4; i < 16; i += 4) {
        env[i + 0] += (env[i - 4] - env[i + 0]) * enva[0];
        env[i + 1] += (env[i - 3] - env[i + 1]) * enva[1];
        env[i + 2] += (env[i - 2] - env[i + 2]) * enva[2];
        env[i + 3] += (env[i - 1] - env[i + 3]) * enva[3];
      }

      Vec resa = (env[i - 4] + env[i - 3] + env[i - 2] - env[i - 1] - env5);

      auto const increasing = resa >= prevr;

      // rel
      envr[0] += (resa - envr[0]) * envb[0];
      envr[1] += (envr5 - envr[1]) * envb[1];
      envr[2] += (envr[4 * 3 + 1] - envr[2]) * envb[2];
      envr[3] += (envr[4 * 3 + 0] - envr[3]) * envb[3];
      envr5 += (envr[4 * 3 + 0] - envr5) * envb5;

      for (i = 4; i < 16; i += 4) {
        envr[i + 0] += (envr[i - 4] - envr[i + 0]) * envb[0];
        envr[i + 1] += (envr[i - 3] - envr[i + 1]) * envb[1];
        envr[i + 2] += (envr[i - 2] - envr[i + 2]) * envb[2];
        envr[i + 3] += (envr[i - 1] - envr[i + 3]) * envb[3];
      }

      prevr = envr[i - 4] + envr[i - 3] + envr[i - 2] - envr[i - 1] - envr5;

      // att
      for (i = 0; i < 16; i += 4) {
        envr[i + 0] = select(increasing, resa, envr[i + 0]);
        envr[i + 1] = select(increasing, resa, envr[i + 1]);
        envr[i + 2] = select(increasing, resa, envr[i + 2]);
        envr[i + 3] = select(increasing, resa, envr[i + 3]);
      }

      envr5 = select(increasing, resa, envr5);
      prevr = select(increasing, resa, prevr);

      if constexpr (outputInDB) {
        return to_db_coef * log(prevr + std::numeric_limits<float>::min());
      }
      else {
        return prevr;
      }
    }
  };

  struct VecDataSymm final
  {
    Vec env[16];
    Vec enva[4];
    Vec env5;
    Vec enva5;
    Mask rms;
    Vec to_db_coef;

    VecDataSymm(GammaEnv const& gammaEnv)
      : env5(Vec().load_a(gammaEnv.env5))
      , enva5(Vec().load_a(gammaEnv.enva5))
      , rms(Vec().load_a(gammaEnv.useRms) != 0.0)
    {
      to_db_coef = select(
        rms, 10.0 / 2.30258509299404568402, 20.0 / 2.30258509299404568402);

      for (int i = 0; i < 4; ++i) {
        enva[i] = Vec().load_a(gammaEnv.enva + i * Vec::size());
      }
      for (int i = 0; i < 16; ++i) {
        env[i] = Vec().load_a(gammaEnv.env + i * Vec::size());
      }
    }

    void update(GammaEnv& gammaEnv) const
    {
      for (int i = 0; i < 4; ++i) {
        enva[i].store_a(gammaEnv.enva + i * Vec::size());
      }
      for (int i = 0; i < 16; ++i) {
        env[i].store_a(gammaEnv.env + i * Vec::size());
      }
      env5.store_a(gammaEnv.env5);
      enva5.store_a(gammaEnv.enva5);
    }

    Vec process(Vec in) { return process_<0>(in); }

    Vec processDB(Vec in) { return process_<1>(in); }

    template<int outputInDB = 0>
    Vec process_(Vec in)
    {
      in = select(rms, in * in, abs(in));
      env[0] += (in - env[0]) * enva[0];
      env[1] += (env5 - env[1]) * enva[1];
      env[2] += (env[4 * 3 + 1] - env[2]) * enva[2];
      env[3] += (env[4 * 3 + 0] - env[3]) * enva[3];
      env5 += (env[4 * 3 + 0] - env5) * enva5;
      int i;

      for (i = 4; i < 16; i += 4) {
        env[i + 0] += (env[i - 4] - env[i + 0]) * enva[0];
        env[i + 1] += (env[i - 3] - env[i + 1]) * enva[1];
        env[i + 2] += (env[i - 2] - env[i + 2]) * enva[2];
        env[i + 3] += (env[i - 1] - env[i + 3]) * enva[3];
      }

      Vec out = (env[i - 4] + env[i - 3] + env[i - 2] - env[i - 1] - env5);

      if constexpr (outputInDB) {
        return to_db_coef * log(out + std::numeric_limits<float>::min());
      }
      else {
        return out;
      }
    }
  };

  GammaEnv() { AVEC_ASSERT_ALIGNMENT(this, Vec); }

  VecData getVecData() const { return VecData(*this); }

  VecDataSymm getVecDataSymm() const { return VecDataSymm(*this); }

  void reset(double initv = 0.0)
  {
    std::fill_n(&env[0], 16 * Vec::size(), initv);
    std::fill_n(&envr[0], 16 * Vec::size(), initv);
    std::fill_n(&env5[0], Vec::size(), initv);
    std::fill_n(&envr5[0], Vec::size(), initv);
    std::fill_n(&prevr[0], Vec::size(), initv);
  }

  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int numSamples)
  {
    processBlock_<0>(input, output, numSamples);
  }

  void processBlockSymm(VecBuffer<Vec> const& input,
                        VecBuffer<Vec>& output,
                        int numSamples)
  {
    processBlockSymm_<0>(input, output, numSamples);
  }

  void processBlockDB(VecBuffer<Vec> const& input,
                      VecBuffer<Vec>& output,
                      int numSamples)
  {
    processBlock_<1>(input, output, numSamples);
  }

  void processBlockSymmDB(VecBuffer<Vec> const& input,
                          VecBuffer<Vec>& output,
                          int numSamples)
  {
    processBlockSymm_<1>(input, output, numSamples);
  }

private:
  template<int outputInDB = 0>
  void processBlock_(VecBuffer<Vec> const& input,
                     VecBuffer<Vec>& output,
                     int numSamples)
  {
    auto v = VecData(*this);

    for (int s = 0; s < numSamples; ++s) {
      output[i] = v.process<outputInDB>(inputs[s]);
    }

    v.update(*this);
  }

  template<int outputInDB = 0>
  void processBlockSymm_(VecBuffer<Vec> const& input,
                         VecBuffer<Vec>& output,
                         int numSamples)
  {
    auto v = VecDataSymm(*this);

    for (int s = 0; s < numSamples; ++s) {
      output[i] = v.process<outputInDB>(inputs[s]);
    }

    v.update(*this);
  }
};

template<class Vec>
class GammaEnvSettings
{
  struct ChannelSettings
  {

    double Attack = 0.0;       ///< Attack frequency.
    double Release = 0.0;      ///< Release frequency.
    double AttackDelay = 0.0;  ///< Attack's delay stage percentage [0; 0.25].
    double ReleaseDelay = 0.0; ///< Release's delay stage percentage [0; 0.25].

    double enva[4]; ///< Attack stage envelope multipliers 1-4.
    double envb[4]; ///< Release stage envelope multipliers 1-4.
    double enva5;   ///< Attack stage envelope multiplier 5.
    double envb5;   ///< Release stage envelope multiplier 5.

    /**
     * Function calculates low-pass filter coefficients (multipliers) for the
     * specified SampleRate, Time and o values. This function's implementation
     * is based on a set of tabulated values transformed into formulas. Hence
     * it may not be useful to explore this function, because the original
     * tabulated values were auto-generated via non-linear optimization: while
     * these values are useful (they just work), they are not descriptive of
     * the underlying regularity.
     *
     * @param Frequency Envelope's frequency.
     * @param o Envelope's delay in percent [0; 0.25].
     * @param[out] envs Resulting envelope multipliers 1-4.
     * @param[out] envs5 Resulting envelope multiplier 5.
     */

    static void calcMults(const double Frequency,
                          const double o,
                          double* const envs,
                          double& envs5)
    {
      const double o2 = o * o;

      if (o <= 0.074) {
        envs[3] = 0.44548 + 0.00920770 * cos(90.2666 * o) - 3.18551 * o -
                  0.132021 * cos(377.561 * o2) -
                  90.2666 * o * o2 * cos(90.2666 * o);
      }
      else if (o <= 0.139) {
        envs[3] = 0.00814353 + 3.07059 * o + 0.00356226 * cos(879.555 * o2);
      }
      else if (o <= 0.180) {
        envs[3] = 0.701590 + o2 * (824.473 * o * o2 - 11.8404);
      }
      else {
        envs[3] = 1.86814 + o * (84.0061 * o2 - 10.8637) - 0.0122863 / o2;
      }

      const double e3 = envs[3];

      envs[0] = 0.901351 +
                o * (12.2872 * e3 + o * (78.0614 - 213.130 * o) - 9.82962) +
                e3 * (0.024808 * exp(7.29048 * e3) - 5.4571 * e3);
      const double e0 = envs[0];

      const double e3exp = exp(1.31354 * e3 + 0.181498 * o);
      envs[1] = e3 * (e0 * (2.75054 * o - 1.0) - 0.611813 * e3 * e3exp) +
                0.821369 * e3exp - 0.845698;
      const double e1 = envs[1];

      envs[2] = 0.860352 + e3 * (1.17208 - 0.579576 * e0) +
                o * (e0 * (1.94324 - 1.95438 * o) + 1.20652 * e3) -
                1.08482 * e0 - 2.14670 * e1;

      if (o >= 0.0750) {
        envs5 = 0.00118;
      }
      else {
        envs5 = e0 * (2.68318 - 2.08720 * o) + 0.485294 * log(e3) +
                3.5805e-10 * exp(27.0504 * e0) - 0.851199 - 1.24658 * e3 -
                0.885938 * log(e0);
      }

      envs[0] = calcLP1CoeffLim(Frequency / envs[0]);
      envs[1] = calcLP1CoeffLim(Frequency / envs[1]);
      envs[2] = calcLP1CoeffLim(Frequency / envs[2]);
      envs[3] = calcLP1CoeffLim(Frequency / envs[3]);
      envs5 = calcLP1CoeffLim(Frequency / envs5);
    }

    /**
     * Function calculates first-order low-pass filter coefficient for the
     * given Theta frequency (0 to pi, inclusive). Returned coefficient in the
     * form ( 1.0 - coeff ) can be used as a coefficient for a high-pass
     * filter. This ( 1.0 - coeff ) can be also used as a gain factor for the
     * high-pass filter so that when high-passed signal is summed with
     * low-passed signal at the same Theta frequency the resuling sum signal
     * is unity.
     *
     * @param theta Low-pass filter's circular frequency, >= 0.
     */

    static double calcLP1Coeff(const double theta)
    {
      const double costheta2 = 2.0 - cos(theta);
      return (1.0 - (costheta2 - sqrt(costheta2 * costheta2 - 1.0)));
    }

    /**
     * Function checks the supplied parameter, limits it to "pi" and calls the
     * calcLP1Coeff() function.
     *
     * @param theta Low-pass filter's circular frequency, >= 0.
     */

    static double calcLP1CoeffLim(const double theta)
    {
      return (calcLP1Coeff(theta < 3.14159265358979324 ? theta
                                                       : 3.14159265358979324));
    }

  public:
    /**
     * Function initializes or updates the internal variables. All public
     * variables have to be defined before calling this function. The clear()
     * function is needed to be called after the first init() function call.
     */

    void init()
    {
      double a;
      double adly;
      double b;
      double bdly;

      if (Attack < Release) {
        a = Attack;
        b = Release;
        adly = AttackDelay;
        bdly = ReleaseDelay;
      }
      else {
        b = Attack;
        a = Release;
        bdly = AttackDelay;
        adly = ReleaseDelay;
      }

      calcMults(a, adly, enva, enva5);
      calcMults(b, bdly, envb, envb5);
    }
  };

  ChannelSettings settings[Vec::size()];
  GammaEnv<Vec>& processor;

public:
  GammaEnvSettings(GammaEnv<Vec>& processor)
    : processor(processor)
  {
    computeCoefficients();
  }

  void computeCoefficients(int channel)
  {
    settings[channel].init();
    for (int s = 0; s < 4; ++s) {
      processor.enva[s * Vec::size() + channel] = settings[channel].enva[s];
      processor.envb[s * Vec::size() + channel] = settings[channel].envb[s];
    }
    processor.enva5[channel] = settings[channel].enva5;
    processor.envb5[channel] = settings[channel].envb5;
  }

  void computeCoefficients()
  {
    for (int c = 0; c < Vec::size(); ++c) {
      computeCoefficients(c);
    }
  }

  void setup(int channel,
             bool rms,
             double Attack,
             double Release,
             double AttackDelay = 0.0,
             double ReleaseDelay = 0.0)
  {
    bool isChanged = false;
    if (settings[channel].Attack != Attack) {
      settings[channel].Attack = Attack;
      isChanged = true;
    }
    if (settings[channel].Release != Release) {
      settings[channel].Release = Release;
      isChanged = true;
    }
    if (settings[channel].AttackDelay != AttackDelay) {
      settings[channel].AttackDelay = AttackDelay;
      isChanged = true;
    }
    if (settings[channel].ReleaseDelay != ReleaseDelay) {
      settings[channel].ReleaseDelay = ReleaseDelay;
      isChanged = true;
    }
    if (isChanged) {
      computeCoefficients(channel);
    }
    processor.useRms[channel] = rms ? 1.0 : 0.0;
  }
};

} // namespace adsp
