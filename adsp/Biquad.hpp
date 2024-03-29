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
#include <array>

namespace adsp {

enum class BiquadFilterType
{
  LowPass = 0,
  HighPass,
  LowShelf,
  HighShelf,
  BandPass,
  Peak,
  Notch,
  AllPass
};

constexpr int numBiquadFilterTypes =
  static_cast<int>(BiquadFilterType::AllPass);

/**
 * An interface for VecBiquadFilter that abstracts over the simd register
 * size. The only methods included in this interface are those to set and get
 * the state and settings of the filter. @see VecBiquadFilter
 */
template<typename Float>
class VecBiquadFilterInterface
{
public:
  virtual void reset(int channel) = 0;
  virtual void reset() = 0;
  virtual void setup(int channel) = 0;
  virtual void setFrequency(int channel, double value) = 0;
  virtual void setFrequency(double value) = 0;
  virtual void setGain(int channel, double value) = 0;
  virtual void setGain(double value) = 0;
  virtual void setQuality(int channel, double value) = 0;
  virtual void setQuality(double value) = 0;
  virtual void setBiquadFilterType(int channel, BiquadFilterType value) = 0;
  virtual void setBiquadFilterType(BiquadFilterType value) = 0;
  virtual double getFrequency(int channel) const = 0;
  virtual double getGain(int channel) const = 0;
  virtual double getQuality(int channel) const = 0;
  virtual BiquadFilterType getBiquadFilterType(int channel) const = 0;
  virtual void setState(int channel, Float state0, Float state1) = 0;
  virtual void getState(int channel, Float& state0, Float& state1) const = 0;
};

/**
 * A simple biquad filter working with VecBuffers.
 */
template<class Vec>
class VecBiquadFilter final
  : public VecBiquadFilterInterface<typename ScalarTypes<Vec>::Float>
{
public:
  using Float = typename ScalarTypes<Vec>::Float;

  /**
   * Constructor.
   * @param filterType_ the type of filter (low pass, high shelf, ... @see
   * BiquadFilterType)
   * @param frequency_ the cutoff (angular) frequency of the filter
   * @param quality_ the quality (Q) of the filter, must be >= 0.5
   * @param gain_ the gain of the filter - only used by shelves and peaking
   * filter types
   */
  VecBiquadFilter(BiquadFilterType filterType_,
                  double frequency_ = 0.1,
                  double quality_ = 0.79,
                  double gain_ = 0.0)
    : buffer(7 * avec::size<Vec>())
    , isSetupNeeded(avec::size<Vec>(), 0)
  {
    std::fill(filterType.begin(), filterType.end(), filterType_);
    std::fill(frequency.begin(), frequency.end(), frequency_);
    std::fill(quality.begin(), quality.end(), quality_);
    std::fill(gain.begin(), gain.end(), gain_);
    setup(true);
  }

  /**
   * Applies the filter to the input and store the result in the output.
   * @param input the input
   * @param output the output
   * @param numSamples the number of samples to process
   */
  void processBlock(VecBuffer<Vec> const& input,
                    VecBuffer<Vec>& output,
                    int numSamples)
  {
    Vec a1 = buffer[0];
    Vec a2 = buffer[1];
    Vec b0 = buffer[2];
    Vec b1 = buffer[3];
    Vec b2 = buffer[4];
    Vec prev0 = buffer[5];
    Vec prev1 = buffer[6];

    for (int i = 0; i < numSamples; ++i) {
      Vec in = input[i];
      Vec next_buffer_0 = in - a1 * prev0 - a2 * prev1;
      Vec out = b0 * next_buffer_0 + b1 * prev0 + b2 * prev1;
      prev1 = prev0;
      prev0 = next_buffer_0;
      output[i] = out;
    }
    buffer[5] = prev0;
    buffer[6] = prev1;
  }

  /**
   * Resets the state of the filter, as if it was processing silence from an
   * eternity.
   */
  void reset() override
  {
    buffer[5] = 0.0;
    buffer[6] = 0.0;
  }

  /**
   * Resets the state of the filter for the specified channel, as if it was
   * processing silence from an eternity.
   * @param channel the channel to reset
   */
  void reset(int channel) override
  {
    buffer[5][channel] = 0.0;
    buffer[6][channel] = 0.0;
  }

  /**
   * Sets the cutoff frequency on a specific channel
   * @param channel the channel on which to change the cutoff frequency
   * @param value the new cutoff frequency
   */
  void setFrequency(int channel, double value) override
  {
    if (frequency[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
    frequency[channel] = value;
  }

  /**
   * Sets the gain on a specific channel
   * @param channel the channel on which to change the gain
   * @param value the new gain
   */
  void setGain(int channel, double value) override
  {
    if (gain[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
    gain[channel] = value;
  }

  /**
   * Sets the quality on a specific channel
   * @param channel the channel on which to change the quality
   * @param value the new quality
   * will be computed. Otherwise they will not be computed.
   */
  void setQuality(int channel, double value) override
  {
    if (quality[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
    quality[channel] = value;
  }

  /**
   * Sets the filter type on a specific channel
   * @param channel the channel on which to change the filter type
   * @param value the new filter type
   */
  void setBiquadFilterType(int channel, BiquadFilterType value) override
  {
    if (filterType[channel] == value) {
      return;
    }
    isSetupNeeded[channel] = true;
    filterType[channel] = value;
  }

  /**
   * Sets the cutoff frequency on all the channels to the specified value.
   * @param value the new cutoff frequency
   */
  void setFrequency(double value) override
  {
    for (int i = 0; i < avec::size<Vec>(); ++i) {
      isSetupNeeded[i] = frequency[i] != value;
    }
    std::fill(frequency.begin(), frequency.end(), value);
  }

  /**
   * Sets the gain on all the channels to the specified value.
   * @param value the new gain
   */
  void setGain(double value) override
  {
    for (int i = 0; i < avec::size<Vec>(); ++i) {
      isSetupNeeded[i] = gain[i] != value;
    }
    std::fill(gain.begin(), gain.end(), value);
  }

  /**
   * Sets the quality on all the channels to the specified value.
   * @param value the new quality
   */
  void setQuality(double value) override
  {
    for (int i = 0; i < avec::size<Vec>(); ++i) {
      isSetupNeeded[i] = quality[i] != value;
    }
    std::fill(quality.begin(), quality.end(), value);
  }

  /**
   * Sets the filter type on all the channels to the specified value.
   * @param value the new filter type
   */
  void setBiquadFilterType(BiquadFilterType value = true) override
  {
    for (int i = 0; i < avec::size<Vec>(); ++i) {
      isSetupNeeded[i] = filterType[i] != value;
    }
    std::fill(filterType.begin(), filterType.end(), value);
  }

  /**
   * Gets the cutoff frequency used for the specified channel.
   * @param channel the channel to get the cutoff frequency from
   * @return the cutoff frequency of the specified channel.
   */
  double getFrequency(int channel) const override { return frequency[channel]; }

  /**
   * Gets the gain used for the specified channel.
   * @param channel the channel to get the gain from
   * @return the gain of the specified channel.
   */
  double getGain(int channel) const override { return gain[channel]; }

  /**
   * Gets the quality used for the specified channel.
   * @param channel the channel to get the quality from
   * @return the quality of the specified channel.
   */
  double getQuality(int channel) const override { return quality[channel]; }

  /**
   * Gets the filter type used for the specified channel.
   * @param channel the channel to get the filter type from
   * @return the filter type of the specified channel.
   */
  BiquadFilterType getBiquadFilterType(int channel) const override
  {
    return filterType[channel];
  }

  /**
   * Computes the filter coefficients for a specified channel.
   * @param channel the channel to compute the coefficients the coeffcient for
   */
  void setup(int channel) override
  {
    isSetupNeeded[channel] = false;

    double a1 = 0.0;
    double a2 = 0.0;
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double g = pow(10.0, gain[channel] / 40.0);
    double cs = cos(frequency[channel]);
    double sn = sin(frequency[channel]);
    double alpha = sn / (2.0 * quality[channel]);
    double sq = 2.0 * sqrt(g) * alpha;
    switch (filterType[channel]) {
      case BiquadFilterType::HighShelf: {
        double a0inv = 1.0 / ((g + 1.0) - (g - 1.0) * cs + sq);
        b0 = a0inv * g * ((g + 1.0) + (g - 1.0) * cs + sq);
        b1 = -2.0 * a0inv * g * ((g - 1.0) + (g + 1.0) * cs);
        b2 = a0inv * g * ((g + 1.0) + (g - 1.0) * cs - sq);
        a1 = 2.0 * a0inv * ((g - 1.0) - (g + 1.0) * cs);
        a2 = a0inv * ((g + 1.0) - (g - 1.0) * cs - sq);
      } break;
      case BiquadFilterType::HighPass: {
        double a0inv = 1.0 / (1.0 + alpha);
        b0 = a0inv * 0.5 * (1.0 + cs);
        b1 = a0inv * (-1.0 - cs);
        b2 = a0inv * 0.5 * (1.0 + cs);
        a1 = -2.0 * a0inv * cs;
        a2 = (1.0 - alpha) * a0inv;
      } break;
      case BiquadFilterType::LowShelf: {
        double a0inv = 1.0 / ((g + 1.0) + (g - 1.0) * cs + sq);
        b0 = a0inv * g * ((g + 1.0) - (g - 1.0) * cs + sq);
        b1 = 2.0 * a0inv * g * ((g - 1.0) - (g + 1.0) * cs);
        b2 = a0inv * g * ((g + 1.0) - (g - 1.0) * cs - sq);
        a1 = -2.0 * a0inv * ((g - 1.0) + (g + 1.0) * cs);
        a2 = a0inv * ((g + 1.0) + (g - 1.0) * cs - sq);
      } break;
      case BiquadFilterType::LowPass: {
        double a0inv = 1.0 / (1.0 + alpha);
        b0 = a0inv * (1.0 - cs) * 0.5;
        b1 = a0inv * (1.0 - cs);
        b2 = a0inv * (1.0 - cs) * 0.5;
        a1 = a0inv * (-2.0 * cs);
        a2 = a0inv * (1.0 - alpha);
      } break;
      case BiquadFilterType::Peak: {
        double a0inv = 1.0 / (1.0 + alpha / g);
        b0 = a0inv * (1.0 + alpha * g);
        b1 = a0inv * (-2.0 * cs);
        b2 = a0inv * (1.0 - alpha * g);
        a1 = a0inv * (-2.0 * cs);
        a2 = a0inv * (1.0 - alpha / g);
      } break;
      case BiquadFilterType::BandPass: {
        double a0inv = 1.0 / (1.0 + alpha);
        b0 = a0inv * alpha;
        b1 = 0.0;
        b2 = -a0inv * alpha;
        a1 = -2.0 * a0inv * cs;
        a2 = a0inv * (1.0 - alpha);
      } break;
      case BiquadFilterType::Notch: {
        double a0inv = 1.0 / (1.0 + alpha);
        b0 = a0inv;
        b1 = -2.0 * a0inv * cs;
        b2 = a0inv;
        a1 = -2.0 * a0inv * cs;
        a2 = a0inv * (1.0 - alpha);
      } break;
      case BiquadFilterType::AllPass: {
        double a0inv = 1.0 / (1.0 + alpha);
        b0 = a0inv * (1.0 - alpha);
        b1 = -2.0 * a0inv * cs;
        b2 = a0inv * (1.0 + alpha);
        a1 = -2.0 * a0inv * cs;
        a2 = a0inv * (1.0 - alpha);
      } break;
      default:
        assert(false);
    }

    buffer[0][channel] = a1;
    buffer[1][channel] = a2;
    buffer[2][channel] = b0;
    buffer[3][channel] = b1;
    buffer[4][channel] = b2;

    reset(channel);
  }

  /**
   * Computes the filter coefficients.
   */
  void setup()
  {
    for (int i = 0; i < avec::size<Vec>(); ++i) {
      setup(i);
    }
  }

  /**
   * Sets the state of the filter for a specified channel.
   * @param channel the channel for which to set the state of the filter
   * @param state0 the first number of the state of the filter
   * @param state1 the second number of the state of the filter
   */
  void setState(int channel, Float state0, Float state1) override
  {
    buffer[5][channel] = state0;
    buffer[6][channel] = state1;
  }

  /**
   * Gets the state of the filter for a specified channel.
   * @param channel the channel for which to set the state of the filter
   * @param state0 the first number of the state of the filter
   * @param state1 the second number of the state of the filter
   */
  void getState(int channel, Float& state0, Float& state1) const override
  {
    state0 = buffer[5][channel];
    state1 = buffer[6][channel];
  }

  /**
   * Computes any the coefficients necessary for the computation that was not
   * already computed. Usefull if you call setters with
   */
  void makeReady()
  {
    for (int i = 0; i < avec::size<Vec>(); ++i) {
      if (isSetupNeeded[i]) {
        setup(i);
      }
    }
  }

private:
  std::vector<int> isSetupNeeded;
  VecBuffer<Vec> buffer;
  std::array<double, avec::size<Vec>()> frequency;
  std::array<double, avec::size<Vec>()> quality;
  std::array<double, avec::size<Vec>()> gain;
  std::array<BiquadFilterType, avec::size<Vec>()> filterType;
};

/**
 * A simple biquad filter working with InterleavedBuffers.
 */
template<typename Float>
class BiquadFilter final
{
  using Vec8 = typename SimdTypes<Float>::Vec8;
  using Vec4 = typename SimdTypes<Float>::Vec4;
  using Vec2 = typename SimdTypes<Float>::Vec2;
  static constexpr bool VEC8_AVAILABLE = SimdTypes<Float>::VEC8_AVAILABLE;
  static constexpr bool VEC4_AVAILABLE = SimdTypes<Float>::VEC4_AVAILABLE;
  static constexpr bool VEC2_AVAILABLE = SimdTypes<Float>::VEC2_AVAILABLE;

  int numChannels;

  std::vector<VecBiquadFilter<Vec8>> filters8;
  std::vector<VecBiquadFilter<Vec4>> filters4;
  std::vector<VecBiquadFilter<Vec2>> filters2;

  template<class Action, typename ValueType>
  ValueType onChannel(Action action, int channel)
  {
    if constexpr (VEC8_AVAILABLE) {
      if (filters4.size() > 0) {
        if (channel < 4) {
          return action(static_cast<VecBiquadFilterInterface*>(&filters4[0]),
                        channel);
        }
        else {
          auto d8 = std::div(channel - 4, 8);
          return action(
            static_cast<VecBiquadFilterInterface*>(&filters8[d8.quot]), d8.rem);
        }
      }
      else {
        auto d8 = std::div(channel, 8);
        return action(
          static_cast<VecBiquadFilterInterface*>(&filters8[d8.quot]), d8.rem);
      }
    }
    else if constexpr (VEC4_AVAILABLE) {
      if constexpr (VEC2_AVAILABLE) {
        if (filters2.size() > 0) {
          if (channel < 2) {
            return action(static_cast<VecBiquadFilterInterface*>(&filters2[0]),
                          channel;
          }
          else {
            auto d4 = std::div(channel - 2, 4);
            return action(
              static_cast<VecBiquadFilterInterface*>(&filters4[d4.quot]),
              d4.rem);
          }
        }
        else {
          auto d4 = std::div(channel, 4);
          return action(
            static_cast<VecBiquadFilterInterface*>(&filters4[d4.quot]), d4.rem);
        }
      }
      else {
        auto d4 = std::div(channel, 4);
        return action(
          static_cast<VecBiquadFilterInterface*>(&filters4[d4.quot]), d4.rem);
      }
    }
    else {
      auto d2 = std::div(channel, 2);
      return action(static_cast<VecBiquadFilterInterface*>(&filters2[d2.quot]),
                    d2.rem);
    }
  }

public:
  /**
   * Constructor.
   * @param numChannels the number of channels to allocate resources for
   * @param filterType_ the type of filter (low pass, high shelf, ... @see
   * BiquadFilterType)
   * @param frequency_ the cutoff (angular) frequency of the filter
   * @param quality_ the quality (Q) of the filter, must be >= 0.5
   * @param gain_ the gain of the filter - only used by shelves and peaking
   * filter types
   */
  BiquadFilter(int numChannels,
               BiquadFilterType filterType_ = BiquadFilterType::LowPass,
               double frequency_ = 0.1,
               double quality_ = 0.79,
               double gain_ = 0.0)
    : numChannels(numChannels)
  {
    int num2, num4, num8;
    avec::getNumOfVecBuffersUsedByInterleavedBuffer<Float>(
      numChannels, num2, num4, num8);
    filters8.reserve(num8);
    filters4.reserve(num4);
    filters2.reserve(num2);
    for (int i = 0; i < num8; ++i) {
      filters8.push_back(
        VecBiquadFilter<Vec8>(filterType_, frequency_, quality_, gain_));
    }
    for (int i = 0; i < num4; ++i) {
      filters4.push_back(
        VecBiquadFilter<Vec4>(filterType_, frequency_, quality_, gain_));
    }
    for (int i = 0; i < num2; ++i) {
      filters2.push_back(
        VecBiquadFilter<Vec2>(filterType_, frequency_, quality_, gain_));
    }
  }

  /**
   * Applies the filter to the input and store the result in the output.
   * @param input the input
   * @param output the output
   * @param numSamples the number of samples to process
   * @param numChannelsToProcess the number of channels to process
   */
  void processBlock(InterleavedBuffer<Float> const& input,
                    InterleavedBuffer<Float>& output,
                    int numSamples,
                    int numChannelsToProcess)
  {
    assert(numChannelsToProcess <= numChannels);
    assert(numSamples <= input.getNumSamples());

    int channelsCount = numChannelsToProcess;

    output.setNumSamples(numSamples);

    if constexpr (VEC2_AVAILABLE) {
      int lastBuffers2 = output.getNumBuffers2() - 1;
      for (int i = 0; i < filters2.size(); ++i) {
        filters2[i].processBlock(
          input.getBuffer2(i), output.getBuffer2(i), numSamples);
        channelsCount -= 2;
        if (channelsCount <= 0) {
          break;
        }
      }
    }
    if constexpr (VEC4_AVAILABLE) {
      for (int i = 0; i < filters4.size(); ++i) {
        filters4[i].processBlock(
          input.getBuffer4(i), output.getBuffer4(i), numSamples);
        channelsCount -= 4;
        if (channelsCount <= 0) {
          break;
        }
      }
    }
    if constexpr (VEC8_AVAILABLE) {
      for (int i = 0; i < filters8.size(); ++i) {
        filters8[i].processBlock(
          input.getBuffer8(i), output.getBuffer8(i), numSamples);
        channelsCount -= 8;
        if (channelsCount <= 0) {
          break;
        }
      }
    }
  }

  /**
   * Resets the state of the filter, as if it was processing silence from an
   * eternity.
   */
  void reset()
  {
    for (auto& f : filters8) {
      f.reset();
    }
    for (auto& f : filters4) {
      f.reset();
    }
    for (auto& f : filters2) {
      f.reset();
    }
  }
  /**
   * Resets the state of the filter for the specified channels
   * param firstChannelToReset the first channel to reset
   * param numChannelsToReset the number of channels to reset
   */
  void resetChannnelStates(int firstChannelToReset, int numChannelsToReset)
  {
    for (int i = firstChannelToReset;
         i < std::min((int)filters2.size(),
                      firstChannelToReset + numChannelsToReset);
         ++i) {
      filters2[i].reset();
    }
    for (int i = firstChannelToReset;
         i < std::min((int)filters4.size(),
                      firstChannelToReset + numChannelsToReset);
         ++i) {
      filters4[i].reset();
    }
    for (int i = firstChannelToReset;
         i < std::min((int)filters8.size(),
                      firstChannelToReset + numChannelsToReset);
         ++i) {
      filters8[i].reset();
    }
  }

  /**
   * Moves the state of the filter at the specied channels to other channels
   * param srcChannel the first channel to move from
   * param dstChannel the first to move to
   * param numChannelsToMove the number of channels to move
   */
  void moveChannnelStates(int srcChannel, int dstChannel, int numChannelsToMove)
  {
    assert(srcChannel + numChannelsToMove < numChannels);
    assert(dstChannel + numChannelsToMove < numChannels);
    assert(std::abs(dstChannel - srChannel) <= numChannelsToMove);

    for (int i = 0; i < numChannelsToMove; ++i) {
      Float state0, state1;
      avec::InterleavedChannel<Float>::doAtChannel(
        i + srcChannel,
        filters2,
        filters4,
        filters8,
        [&](auto& filter, int channel, int numChannels) {
          filter.getState(channel, state0, state1);
        });
      avec::InterleavedChannel<Float>::doAtChannel(
        i + dstChannel,
        filters2,
        filters4,
        filters8,
        [&](auto& filter, int channel, int numChannels) {
          filter.setState(channel, state0, state1);
        });
    }
  }

  /**
   * Sets the cutoff frequency on a specific channel
   * @param channel the channel on which to change the cutoff frequency
   * @param value the new cutoff frequency
   * will be computed. Otherwise they will not be computed.
   */
  void setFrequency(int channel, double value)
  {
    avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [value](auto& filter, int channel, int unused_numChannels) {
        filter.setFrequency(channel, value);
      });
  }

  /**
   * Sets the gain on a specific channel
   * @param channel the channel on which to change the gain
   * @param value the new gain
   */
  void setGain(int channel, double value)
  {
    avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [value](auto& filter, int channel, int unused_numChannels) {
        filter.setGain(channel, value);
      });
  }

  /**
   * Sets the quality on a specific channel
   * @param channel the channel on which to change the quality
   * @param value the new quality
   */
  void setQuality(int channel, double value)
  {
    avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [value](auto& filter, int channel, int unused_numChannels) {
        filter.setQuality(channel, value);
      });
  }

  /**
   * Sets the filter type on a specific channel
   * @param channel the channel on which to change the filter type
   * @param value the new filter type
   */
  void setBiquadFilterType(int channel, BiquadFilterType value)
  {
    avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [value](auto& filter, int channel, int unused_numChannels) {
        filter.setBiquadFilterType(channel, value);
      });
  }

  /**
   * Sets the cutoff frequency on all the channels to the specified value.
   * @param value the new cutoff frequency
   */
  void setFrequency(double value)
  {
    for (auto& f : filters8) {
      f.setFrequency(value);
    }
    for (auto& f : filters4) {
      f.setFrequency(value);
    }
    for (auto& f : filters2) {
      f.setFrequency(value);
    }
  }

  /**
   * Sets the gain on all the channels to the specified value.
   * @param value the new gain
   */
  void setGain(double value)
  {
    for (auto& f : filters8) {
      f.setGain(value);
    }
    for (auto& f : filters4) {
      f.setGain(value);
    }
    for (auto& f : filters2) {
      f.setGain(value);
    }
  }

  /**
   * Sets the quality on all the channels to the specified value.
   * @param value the new quality
   */
  void setQuality(double value)
  {
    for (auto& f : filters8) {
      f.setQuality(value);
    }
    for (auto& f : filters4) {
      f.setQuality(value);
    }
    for (auto& f : filters2) {
      f.setQuality(value);
    }
  }

  /**
   * Sets the filter type on all the channels to the specified value.
   * @param value the new filter type
   */
  void setBiquadFilterType(BiquadFilterType value = true)
  {
    for (auto& f : filters8) {
      f.setBiquadFilterType(value);
    }
    for (auto& f : filters4) {
      f.setBiquadFilterType(value);
    }
    for (auto& f : filters2) {
      f.setBiquadFilterType(value);
    }
  }

  /**
   * Gets the cutoff frequency used for the specified channel.
   * @param channel the channel to get the cutoff frequency from
   * @return the cutoff frequency of the specified channel.
   */
  double getFrequency(int channel) const
  {
    return avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [](auto& filter, int channel, int unused_numChannels) {
        return filter.getFrequency(channel);
      });
  }

  /**
   * Gets the gain used for the specified channel.
   * @param channel the channel to get the gain from
   * @return the gain of the specified channel.
   */
  double getGain(int channel) const
  {
    return avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [](auto& filter, int channel, int unused_numChannels) {
        return filter.getGain(channel);
      });
  }

  /**
   * Gets the quality used for the specified channel.
   * @param channel the channel to get the quality from
   * @return the quality of the specified channel.
   */
  double getQuality(int channel) const
  {
    return avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [](auto& filter, int channel, int unused_numChannels) {
        return filter.getQuality(channel);
      });
  }

  /**
   * Gets the filter type used for the specified channel.
   * @param channel the channel to get the filter type from
   * @return the filter type of the specified channel.
   */
  BiquadFilterType getBiquadFilterType(int channel) const
  {
    return avec::InterleavedChannel<Float>::doAtChannel(
      channel,
      filters2,
      filters4,
      filters8,
      [](auto& filter, int channel, int unused_numChannels) {
        return filter.getBiquadFilterType(channel);
      });
  }

  /**
   * @return the maximum number of channel that the generator can work with.
   */
  int getNumChannels() const { return numChannels; }

  /**
   * Computes any the coefficients necessary for the computation that was not
   * already computed. Usefull if you call setters with
   */
  void makeReady()
  {
    for (auto& f : filters8) {
      f.makeReady();
    }
    for (auto& f : filters4) {
      f.makeReady();
    }
    for (auto& f : filters2) {
      f.makeReady();
    }
  }
};

} // namespace adsp