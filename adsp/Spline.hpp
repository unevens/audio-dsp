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

  template<class VecClass, int numKnots, class Automator>
  friend class SplineDispatcher;

  using Scalar = typename ScalarTypes<Vec>::Scalar;
  using Mask = typename MaskTypes<Vec>::Mask;

  struct Knot final
  {
    Scalar x[Vec::size()];
    Scalar y[Vec::size()];
    Scalar t[Vec::size()];
    Scalar s[Vec::size()];
  };

  struct Knots final
  {
    Knot knots[maxNumKnots];
    Knot& operator[](int i) { return knots[i]; }
    Knot const& operator[](int i) const { return knots[i]; }
  };

  struct Settings final
  {
    Scalar isSymmetric[Vec::size()];
    Knots knots;
  };

  struct VecKnot final
  {
    Vec x;
    Vec y;
    Vec t;
    Vec s;

    VecKnot() = default;

    VecKnot(Knot const& knot)
      : x(Vec().load_a(knot.x))
      , y(Vec().load_a(knot.y))
      , t(Vec().load_a(knot.t))
      , s(Vec().load_a(knot.s))
    {}
  };

  template<int maxNumActiveKnots>
  struct VecKnots final
  {
    VecKnot knots[maxNumActiveKnots];

    VecKnots(Knots const& storage)
    {
      for (int i = 0; i < maxNumActiveKnots; ++i) {
        knots[i] = VecKnot(storage[i]);
      }
    }

    VecKnot& operator[](int i) { return knots[i]; }

    VecKnot const& operator[](int i) const { return knots[i]; }

    void update(Knots& storage, int const numActiveKnots) const
    {
      for (int n = 0; n < numActiveKnots; ++n) {
        knots[n].x.store_a(storage[n].x);
        knots[n].y.store_a(storage[n].y);
        knots[n].t.store_a(storage[n].t);
        knots[n].s.store_a(storage[n].s);
      }
    }
  };

  struct SmoothingAutomator final
  {
    Scalar smoothingAlpha[Vec::size()];
    Knots knots;

    template<int maxNumActiveKnots>
    struct VecAutomator
    {
      using Automator = SmoothingAutomator;
      VecKnots<maxNumActiveKnots> knots;
      Vec alpha;

      VecAutomator(SmoothingAutomator const& automator)
        : knots(automator.knots)
        , alpha(Vec().load_a(automator.smoothingAlpha))
      {}

      void automate(VecKnots<maxNumActiveKnots>& values,
                    int const numActiveKnots) const
      {
        for (int n = 0; n < numActiveKnots; ++n) {
          values[n].x = alpha * (values[n].x - knots[n].x) + knots[n].x;
          values[n].y = alpha * (values[n].y - knots[n].y) + knots[n].y;
          values[n].t = alpha * (values[n].t - knots[n].t) + knots[n].t;
          values[n].s = alpha * (values[n].s - knots[n].s) + knots[n].s;
        }
      }
    };

    void setSmoothingAlpha(Scalar alpha)
    {
      std::fill_n(smoothingAlpha, Vec::size(), alpha);
    }

    void reset(Spline& spline)
    {
      std::copy(&knots[0], &knots[0] + maxNumKnots, &spline.settings.knots[0]);
    }

    template<int maxNumActiveKnots>
    VecAutomator<maxNumActiveKnots> getVecAutomator() const
    {
      return VecAutomator<maxNumActiveKnots>(*this);
    }
  };

  struct FakeAutomator final
  {
    template<int maxNumActiveKnots>
    struct VecAutomator
    {
      using Automator = FakeAutomator;
      void automate(VecKnots<maxNumActiveKnots>& values,
                    int const numActiveKnots) const
      {}
    };

    void reset(Spline& spline) const {}

    template<int maxNumActiveKnots>
    VecAutomator<maxNumActiveKnots> getVecAutomator() const
    {
      return {};
    }

    FakeAutomator() = default;
  };

  template<int maxNumActiveKnots>
  struct VecSpline final
  {
    Mask isSymmetric;
    VecKnots<maxNumActiveKnots> knots;

    VecSpline(Settings const& settings)
      : isSymmetric(Vec().load_a(settings.isSymmetric) != 0.0)
      , knots(settings.knots)
    {}

    void update(Spline& spline, int const numActiveKnots)
    {
      knots.update(spline.settings.knots, numActiveKnots);
    }

    template<template<int>
             class AutomatorVecData = FakeAutomator::template VecAutomator>
    Vec process(Vec const input,
                AutomatorVecData<maxNumActiveKnots> const& automation,
                int const numActiveKnots = maxNumActiveKnots);
  };

  Settings settings;

  Spline()
  {
    AVEC_ASSERT_ALIGNMENT(this, Vec);
    std::fill_n(settings.isSymmetric, (4 * maxNumKnots + 1) * Vec::size(), 0.0);
  }

  void setIsSymmetric(bool value)
  {
    std::fill_n(settings.isSymmetric, Vec::size(), value ? 1.0 : 0.0);
  }

  void setIsSymmetric(int channel, bool value)
  {
    settings.isSymmetric[channel] = value ? 1.0 : 0.0;
  }

  template<int numActiveKnots = maxNumKnots, class Automator = FakeAutomator>
  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    Automator const& automator = {})
  {
    processBlock_<numActiveKnots, Automator>(
      input, output, numActiveKnots, automator);
  }

  template<class Automator = FakeAutomator>
  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int const numActiveKnots,
                    Automator const& automator = {})
  {
    processBlock_<maxNumKnots, Automator>(
      input, output, numActiveKnots, automator);
  }

  template<int maxNumActiveKnots>
  void update(VecSpline<maxNumActiveKnots> const& spline,
              int const numActiveKnots)
  {
    spline.knots.update(settings.knots, numActiveKnots);
  }

  template<int maxNumActiveKnots>
  VecSpline<maxNumActiveKnots> getVecSpline() const
  {
    return VecSpline<maxNumActiveKnots>(settings);
  }

private:
  template<int maxNumActiveKnots, class Automator = FakeAutomator>
  void processBlock_(VecBuffer<Vec> const& input,
                     VecBuffer<Vec>& output,
                     int const numActiveKnots,
                     Automator const& automator = {});
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
  using Automator = typename Spline<Vec, maxNumKnots>::SmoothingAutomator;

  Spline<Vec, maxNumKnots> spline;
  Automator automator;

  void reset() { automator.reset(spline); }

  template<int numActiveKnots = maxNumKnots>
  void processBlock(VecBuffer<Vec> const& input, VecBuffer<Vec>& output)
  {
    spline.template processBlock<numActiveKnots, Automator>(
      input, output, automator);
  }

  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int const numActiveKnots)
  {
    spline.template processBlock<Automator>(
      input, output, numActiveKnots, automator);
  }
};

/**
 * A class to use the version of the Spline::processBlock method which takes the
 * number of active knots as a template argument even when the number of
 * knots is not known at compile time.
 */
template<class Vec,
         int maxNumKnots,
         class Automator = typename Spline<Vec, maxNumKnots>::FakeAutomator>
class SplineDispatcher final
{
  using Call = void (Spline<Vec, maxNumKnots>::*)(VecBuffer<Vec> const&,
                                                  VecBuffer<Vec>&,
                                                  int const,
                                                  Automator const&);

  std::array<Call, maxNumKnots + 1> calls;

  template<int numActiveKnots>
  struct Initializer
  {
    static void initialize(Call* calls)
    {
      calls[numActiveKnots] =
        &Spline<Vec, maxNumKnots>::template processBlock_<numActiveKnots,
                                                          Automator>;
      if constexpr (numActiveKnots > 0) {
        Initializer<numActiveKnots - 1>::initialize(calls);
      }
    }
  };

public:
  SplineDispatcher() { Initializer<maxNumKnots>::initialize(&calls[0]); }

  void processBlock(Spline<Vec, maxNumKnots>& spline,
                    VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int const numActiveKnots,
                    Automator const& automator = {})
  {
    (spline.*(calls[numActiveKnots]))(input, output, numActiveKnots, automator);
  }
};

/**
 * A class to use the version of the AutoSpline::processBlock method which takes
 * the number of active knots as a template argument even when the number of
 * knots is not known at compile time.
 */
template<class Vec, int maxNumKnots>
class AutoSplineDispatcher final
{
  SplineDispatcher<Vec,
                   maxNumKnots,
                   typename Spline<Vec, maxNumKnots>::SmoothingAutomator>
    dispatcher;

public:
  void processBlock(AutoSpline<Vec, maxNumKnots>& spline,
                    VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int const numActiveKnots)
  {
    dispatcher.processBlock(
      spline.spline, input, output, numActiveKnots, spline.automator);
  }
};

// implementation

template<class Vec, int maxNumKnots_>
template<int maxNumActiveKnots_, class Automator>
inline void
Spline<Vec, maxNumKnots_>::processBlock_(VecBuffer<Vec> const& input,
                                         VecBuffer<Vec>& output,
                                         int const numActiveKnots,
                                         Automator const& automator)

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

  auto spline = getVecSpline<maxNumActiveKnots>();
  auto automation = automator.template getVecAutomator<maxNumActiveKnots>();

  for (int i = 0; i < numSamples; ++i) {
    output[i] = spline.template process<Automator::template VecAutomator>(
      input[i], automation, numActiveKnots);
  }

  if constexpr (!std::is_same_v<Automator, FakeAutomator>) {
    spline.update(*this, numActiveKnots);
  }
}

template<class Vec, int maxNumKnots_>
template<int maxNumActiveKnots>
template<template<int> class AutomatorVecData>
inline Vec
Spline<Vec, maxNumKnots_>::VecSpline<maxNumActiveKnots>::process(
  Vec const input,
  AutomatorVecData<maxNumActiveKnots> const& automation,
  int const numActiveKnots)
{
  Vec const in = select(isSymmetric, abs(input), input);

  if constexpr (!std::is_same_v<
                  typename AutomatorVecData<maxNumActiveKnots>::Automator,
                  FakeAutomator>) {
    automation.automate(knots, numActiveKnots);
  }

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

  Vec x_low = knots[0].x;
  Vec y_low = knots[0].y;
  Vec t_low = knots[0].t;

  // parameters for segment above the range of the spline

  Vec x_high = knots[0].x;
  Vec y_high = knots[0].y;
  Vec t_high = knots[0].t;

  // find interval and set left and right knot parameters

  for (int n = 0; n < numActiveKnots; ++n) {
    auto const is_left = (in > knots[n].x) && (knots[n].x > x0);
    x0 = select(is_left, knots[n].x, x0);
    y0 = select(is_left, knots[n].y, y0);
    t0 = select(is_left, knots[n].t, t0);
    s0 = select(is_left, knots[n].s, s0);

    auto const is_right = (in <= knots[n].x) && (knots[n].x < x1);
    x1 = select(is_right, knots[n].x, x1);
    y1 = select(is_right, knots[n].y, y1);
    t1 = select(is_right, knots[n].t, t1);
    s1 = select(is_right, knots[n].s, s1);

    auto const is_lowest = knots[n].x < x_low;
    x_low = select(is_lowest, knots[n].x, x_low);
    y_low = select(is_lowest, knots[n].y, y_low);
    t_low = select(is_lowest, knots[n].t, t_low);

    auto const is_highest = knots[n].x > x_high;
    x_high = select(is_highest, knots[n].x, x_high);
    y_high = select(is_highest, knots[n].y, y_high);
    t_high = select(is_highest, knots[n].t, t_high);
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

  return select(isSymmetric, sign_combine(out, input), out);
}

} // namespace adsp
