/*
Copyright 2020 Dario Mambro

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

/**
 * A class to evaluate a cubic Hermite spline with a given number of knots.
 */
template<class Vec, int maxNumKnots_>
struct Spline final
{
  static constexpr int maxNumKnots = maxNumKnots_;

  using Scalar = typename ScalarTypes<Vec>::Scalar;

  struct Knot final
  {
    Scalar x[Vec::size()];
    Scalar y[Vec::size()];
    Scalar t[Vec::size()];
    Scalar s[Vec::size()];
  };

  Scalar isSymmetric[Vec::size()];
  Knot knots[maxNumKnots];

  Spline()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    std::fill_n(isSymmetric, (4 * maxNumKnots + 1) * Vec::size(), 0.0);
  }

  void setIsSymmetric(bool value)
  {
    std::fill_n(isSymmetric, Vec::size(), value ? 1.0 : 0.0);
  }

  void setIsSymmetric(int channel, bool value)
  {
    isSymmetric[channel] = value ? 1.0 : 0.0;
  }

  template<int numActiveKnots = maxNumKnots>
  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    processBlock_<numActiveKnots>(input, output, numActiveKnots);
  }

  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int const numActiveKnots)
  {
    processBlock_<maxNumKnots>(input, output, numActiveKnots);
  }

  template<int maxNumActiveKnots>
  void processBlock_(VecBuffer<Vec> const& input,
                     VecBuffer<Vec>& output,
                     int const numActiveKnots);
};

/**
 * A class to evaluate a cubic Hermite spline with a given number of knots whose
 * values are smoothly automated.
 */
template<class Vec, int maxNumKnots_>
struct AutoSpline final
{
  static constexpr int maxNumKnots = maxNumKnots_;

  using Scalar = typename ScalarTypes<Vec>::Scalar;
  using Knot = typename Spline<Vec, maxNumKnots>::Knot;

  Scalar smoothingAlpha[Vec::size()];
  Knot automationKnots[maxNumKnots];

  Spline<Vec, maxNumKnots> spline;

  AutoSpline()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    std::fill_n(smoothingAlpha, (4 * maxNumKnots + 1) * Vec::size(), 0.0);
  }

  void setSmoothingAlpha(Scalar alpha)
  {
    std::fill_n(smoothingAlpha, Vec::size(), alpha);
  }

  void reset()
  {
    std::copy(
      &automationKnots[0], &automationKnots[0] + maxNumKnots, &spline.knots[0]);

    for (int i = 0; i < maxNumKnots; ++i) {
      for (int j = 0; j < Vec::size(); ++j) {
        assert(spline.knots[i].x[j] == automationKnots[i].x[j]);
        assert(spline.knots[i].y[j] == automationKnots[i].y[j]);
        assert(spline.knots[i].s[j] == automationKnots[i].s[j]);
        assert(spline.knots[i].t[j] == automationKnots[i].t[j]);
      }
    }
  }

  template<int numActiveKnots = maxNumKnots>
  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    processBlock_<numActiveKnots>(input, output, numActiveKnots);
  }

  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int const numActiveKnots)
  {
    processBlock_<maxNumKnots>(input, output, numActiveKnots);
  }

  template<int maxNumActiveKnots>
  void processBlock_(VecBuffer<Vec> const& input,
                     VecBuffer<Vec>& output,
                     int const numActiveKnots);
};

/**
 * A class to use the version of the Spline::processBlock method which takes the
 * number of active knots as a template argument even when the number of
 * knots is not known at compile time.
 */
template<template<class, int> class SplineClass, class Vec, int maxNumKnots>
class SplineDispatcher final
{
  using Call = void (SplineClass<Vec, maxNumKnots>::*)(VecBuffer<Vec> const&,
                                                       VecBuffer<Vec>&,
                                                       int const);

  std::array<Call, maxNumKnots + 1> calls;

  template<int numActiveKnots>
  struct Initializer
  {
    static void initialize(Call* calls)
    {
      calls[numActiveKnots] =
        &SplineClass<Vec, maxNumKnots>::template processBlock_<numActiveKnots>;
      if constexpr (numActiveKnots > 0) {
        Initializer<numActiveKnots - 1>::initialize(calls);
      }
    }
  };

public:
  SplineDispatcher() { Initializer<maxNumKnots>::initialize(&calls[0]); }

  void processBlock(SplineClass<Vec, maxNumKnots>& spline,
                    VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int const numActiveKnots)
  {
    (spline.*(calls[numActiveKnots]))(input, output, numActiveKnots);
  }
};

// implementation

template<class Vec, int maxNumKnots_>
template<int maxNumActiveKnots_>
inline void
Spline<Vec, maxNumKnots_>::processBlock_(VecBuffer<Vec> const& input,
                                         VecBuffer<Vec>& output,
                                         int const numActiveKnots)
{
  constexpr int maxNumActiveKnots = std::max(1, maxNumActiveKnots_);

  static_assert(maxNumActiveKnots <= maxNumKnots,
                "maxNumActiveKnots must be less or equal to maxNumKnots.");

  assert(numActiveKnots <= maxNumActiveKnots);

  int const numSamples = input.getNumSamples();
  output.setNumSamples(numSamples);

  if (numActiveKnots == 0) {
    if (&input != &output) {
      std::copy(&input(0), &input(0) + input.getScalarSize(), &output(0));
    }
    return;
  }

  Vec x[maxNumActiveKnots];
  Vec y[maxNumActiveKnots];
  Vec t[maxNumActiveKnots];
  Vec s[maxNumActiveKnots];

  auto symm = Vec().load_a(isSymmetric) != 0.0;

  for (int n = 0; n < numActiveKnots; ++n) {
    x[n] = Vec().load_a(knots[n].x);
    y[n] = Vec().load_a(knots[n].y);
    t[n] = Vec().load_a(knots[n].t);
    s[n] = Vec().load_a(knots[n].s);
  }

  for (int i = 0; i < numSamples; ++i) {

    Vec const in_signed = input[i];

    Vec const in = select(symm, abs(in_signed), in_signed);

    // left knot paramters

    Vec x0 = std::numeric_limits<float>::lowest();
    Vec y0 = 0.f;
    Vec t0 = 0.f;
    Vec s0 = 0.f;

    // right knot paramters

    Vec x1 = std::numeric_limits<float>::max();
    Vec y1 = 0.f;
    Vec t1 = 0.f;
    Vec s1 = 0.f;

    // parameters for segment below the range of the spline

    Vec x_low = x[0];
    Vec y_low = y[0];
    Vec t_low = t[0];

    // parameters for segment above the range of the spline

    Vec x_high = x[0];
    Vec y_high = y[0];
    Vec t_high = t[0];

    // find interval and set left and right knot parameters

    for (int n = 0; n < numActiveKnots; ++n) {
      auto const is_left = (in > x[n]) && (x[n] > x0);
      x0 = select(is_left, x[n], x0);
      y0 = select(is_left, y[n], y0);
      t0 = select(is_left, t[n], t0);
      s0 = select(is_left, s[n], s0);

      auto const is_right = (in <= x[n]) && (x[n] < x1);
      x1 = select(is_right, x[n], x1);
      y1 = select(is_right, y[n], y1);
      t1 = select(is_right, t[n], t1);
      s1 = select(is_right, s[n], s1);

      auto const is_lowest = x[n] < x_low;
      x_low = select(is_lowest, x[n], x_low);
      y_low = select(is_lowest, y[n], y_low);
      t_low = select(is_lowest, t[n], t_low);

      auto const is_highest = x[n] > x_high;
      x_high = select(is_highest, x[n], x_high);
      y_high = select(is_highest, y[n], y_high);
      t_high = select(is_highest, t[n], t_high);
    }

    auto const is_high = x1 == std::numeric_limits<float>::max();
    auto const is_low = x0 == std::numeric_limits<float>::lowest();

    // compute spline and segment coeffcients

    Vec const dx = max(x1 - x0, std::numeric_limits<float>::min());
    Vec const dy = y1 - y0;
    Vec const a = t0 * dx - dy;
    Vec const b = -t1 * dx + dy;
    Vec const ix = 1.0 / dx;
    Vec const m = dy * ix;
    Vec const o = y0 - m * x0;

    // compute spline

    Vec const j = (in - x0) * ix;
    Vec const k = 1.0 - j;
    Vec const hermite = k * y0 + j * y1 + j * k * (a * k + b * j);

    // compute segment and interpolate using smoothness

    Vec const segment = m * in + o;
    Vec const smoothness = s1 + k * (s0 - s1);
    Vec const curve = segment + smoothness * (hermite - segment);

    //  the result if the input is outside the spline range

    Vec const low = y_low + (in - x_low) * t_low;
    Vec const high = y_high + (in - x_high) * t_high;

    Vec const out = select(is_high, high, select(is_low, low, curve));

    // symmetry

    output[i] = select(symm, sign_combine(out, in_signed), out);
  }
}

template<class Vec, int maxNumKnots_>
template<int maxNumActiveKnots_>
inline void
AutoSpline<Vec, maxNumKnots_>::processBlock_(VecBuffer<Vec> const& input,
                                             VecBuffer<Vec>& output,
                                             int const numActiveKnots)
{
  constexpr int maxNumActiveKnots = std::max(1, maxNumActiveKnots_);

  static_assert(maxNumActiveKnots <= maxNumKnots,
                "maxNumActiveKnots must be less or equal to maxNumKnots.");

  assert(numActiveKnots <= maxNumActiveKnots);

  int const numSamples = input.getNumSamples();
  output.setNumSamples(numSamples);

  if (numActiveKnots == 0) {
    if (&input != &output) {
      std::copy(&input(0), &input(0) + input.getScalarSize(), &output(0));
    }
    return;
  }

  Vec const alpha = Vec().load_a(smoothingAlpha);

  Vec x[maxNumActiveKnots];
  Vec y[maxNumActiveKnots];
  Vec t[maxNumActiveKnots];
  Vec s[maxNumActiveKnots];

  Vec x_a[maxNumActiveKnots];
  Vec y_a[maxNumActiveKnots];
  Vec t_a[maxNumActiveKnots];
  Vec s_a[maxNumActiveKnots];

  auto symm = Vec().load_a(spline.isSymmetric) != 0.0;

  for (int n = 0; n < numActiveKnots; ++n) {
    x[n] = Vec().load_a(spline.knots[n].x);
    y[n] = Vec().load_a(spline.knots[n].y);
    t[n] = Vec().load_a(spline.knots[n].t);
    s[n] = Vec().load_a(spline.knots[n].s);
  }

  for (int n = 0; n < numActiveKnots; ++n) {
    x_a[n] = Vec().load_a(automationKnots[n].x);
    y_a[n] = Vec().load_a(automationKnots[n].y);
    t_a[n] = Vec().load_a(automationKnots[n].t);
    s_a[n] = Vec().load_a(automationKnots[n].s);
  }

  for (int i = 0; i < numSamples; ++i) {

    // advance automation

    for (int n = 0; n < numActiveKnots; ++n) {
      x[n] = alpha * (x[n] - x_a[n]) + x_a[n];
      y[n] = alpha * (y[n] - y_a[n]) + y_a[n];
      t[n] = alpha * (t[n] - t_a[n]) + t_a[n];
      s[n] = alpha * (s[n] - s_a[n]) + s_a[n];
    }

    Vec const in_signed = input[i];

    Vec const in = select(symm, abs(in_signed), in_signed);

    // left knot paramters

    Vec x0 = std::numeric_limits<float>::lowest();
    Vec y0 = 0.f;
    Vec t0 = 0.f;
    Vec s0 = 0.f;

    // right knot paramters

    Vec x1 = std::numeric_limits<float>::max();
    Vec y1 = 0.f;
    Vec t1 = 0.f;
    Vec s1 = 0.f;

    // parameters for segment below the range of the spline

    Vec x_low = x[0];
    Vec y_low = y[0];
    Vec t_low = t[0];

    // parameters for segment above the range of the spline

    Vec x_high = x[0];
    Vec y_high = y[0];
    Vec t_high = t[0];

    // find interval and set left and right knot parameters

    for (int n = 0; n < numActiveKnots; ++n) {
      auto const is_left = (in > x[n]) && (x[n] > x0);
      x0 = select(is_left, x[n], x0);
      y0 = select(is_left, y[n], y0);
      t0 = select(is_left, t[n], t0);
      s0 = select(is_left, s[n], s0);

      auto const is_right = (in <= x[n]) && (x[n] < x1);
      x1 = select(is_right, x[n], x1);
      y1 = select(is_right, y[n], y1);
      t1 = select(is_right, t[n], t1);
      s1 = select(is_right, s[n], s1);

      auto const is_lowest = x[n] < x_low;
      x_low = select(is_lowest, x[n], x_low);
      y_low = select(is_lowest, y[n], y_low);
      t_low = select(is_lowest, t[n], t_low);

      auto const is_highest = x[n] > x_high;
      x_high = select(is_highest, x[n], x_high);
      y_high = select(is_highest, y[n], y_high);
      t_high = select(is_highest, t[n], t_high);
    }

    auto const is_high = x1 == std::numeric_limits<float>::max();
    auto const is_low = x0 == std::numeric_limits<float>::lowest();

    // compute spline and segment coeffcients

    Vec const dx = max(x1 - x0, std::numeric_limits<float>::min());
    Vec const dy = y1 - y0;
    Vec const a = t0 * dx - dy;
    Vec const b = -t1 * dx + dy;
    Vec const ix = 1.0 / dx;
    Vec const m = dy * ix;
    Vec const o = y0 - m * x0;

    // compute spline

    Vec const j = (in - x0) * ix;
    Vec const k = 1.0 - j;
    Vec const hermite = k * y0 + j * y1 + j * k * (a * k + b * j);

    // compute segment and interpolate using smoothness

    Vec const segment = m * in + o;
    Vec const smoothness = s1 + k * (s0 - s1);
    Vec const curve = segment + smoothness * (hermite - segment);

    //  the result if the input is outside the spline range

    Vec const low = y_low + (in - x_low) * t_low;
    Vec const high = y_high + (in - x_high) * t_high;

    Vec const out = select(is_high, high, select(is_low, low, curve));

    // symmetry

    output[i] = select(symm, sign_combine(out, in_signed), out);
  }

  // update spline state

  for (int n = 0; n < numActiveKnots; ++n) {
    x[n].store_a(spline.knots[n].x);
    y[n].store_a(spline.knots[n].y);
    t[n].store_a(spline.knots[n].t);
    s[n].store_a(spline.knots[n].s);
  }
}

} // namespace adsp
