#include "p6/pipeline.hpp"
#include <cmath>
#include <algorithm>

namespace p6 {

static constexpr double PI = 3.14159265358979323846;

// Nominal center frequencies for force weighting (matches Python force_aggregator.py).
// These are not the actual band-center frequencies but fixed denominators for 1/f weighting.
//   INSTITUTIONAL=0.125 => weight 8x (dominant, slow-moving institutional flow)
//   FUND         =0.375 => weight ~2.7x
//   DAYTRADING   =0.625 => weight ~1.6x
//   HFT          =0.875 => weight ~1.1x (least weighted)
static constexpr double BAND_CENTERS[4] = {0.125, 0.375, 0.625, 0.875};

// Simple O(N²) DFT (rfft style: k = 0 … N/2).
// For N=FFT_WINDOW=256: 256 * 129 ≈ 33k multiply-add pairs — fast enough for hot path.
static void compute_rfft(const double* x, int N,
                         double* real_out, double* imag_out) {
    const int half = N / 2 + 1;
    for (int k = 0; k < half; ++k) {
        double re = 0.0, im = 0.0;
        const double factor = -2.0 * PI * k / N;
        for (int n = 0; n < N; ++n) {
            double angle = factor * n;
            re += x[n] * std::cos(angle);
            im += x[n] * std::sin(angle);
        }
        real_out[k] = re;
        imag_out[k] = im;
    }
}

ForceVector SpectralForce::compute(const std::vector<Order>& trades,
                                   int64_t timestamp_ms) {
    ForceVector fv;
    fv.timestamp_ms = timestamp_ms;

    // Accumulate signed volume delta into circular buffer.
    // ASK fill = buy aggressor = positive delta
    // BID fill = sell aggressor = negative delta
    for (const auto& t : trades) {
        if (t.action == OrderAction::FILL) {
            double delta = (t.side == Side::ASK) ? t.size : -t.size;
            volume_delta_buf_[buf_pos_] = delta;
            buf_pos_ = (buf_pos_ + 1) % FFT_WINDOW;
        }
    }

    // Linearise the circular buffer (oldest sample first)
    double window[FFT_WINDOW];
    for (int i = 0; i < FFT_WINDOW; ++i) {
        window[i] = volume_delta_buf_[(buf_pos_ + i) % FFT_WINDOW];
    }

    const int half = FFT_WINDOW / 2 + 1;
    double real_buf[half], imag_buf[half];
    compute_rfft(window, FFT_WINDOW, real_buf, imag_buf);

    // Band k-ranges (based on fmax=0.5, quantile split matching band_splitter.py):
    //   INSTITUTIONAL : 0   ≤ k ≤ FFT_WINDOW/8   (f ≤ 0.125)
    //   FUND          : FFT_WINDOW/8  < k ≤ FFT_WINDOW/4   (0.125 < f ≤ 0.25)
    //   DAYTRADING    : FFT_WINDOW/4  < k ≤ 3*FFT_WINDOW/8 (0.25  < f ≤ 0.375)
    //   HFT           : 3*FFT_WINDOW/8 < k ≤ half          (f > 0.375)
    const int band_limits[5] = {
        0,
        FFT_WINDOW / 8,
        FFT_WINDOW / 4,
        3 * FFT_WINDOW / 8,
        half
    };

    double total_energy = 0.0;

    for (int b = 0; b < 4; ++b) {
        double energy    = 0.0;
        double net_real  = 0.0;

        for (int k = band_limits[b]; k < band_limits[b + 1] && k < half; ++k) {
            double mag2 = real_buf[k] * real_buf[k] + imag_buf[k] * imag_buf[k];
            energy   += mag2;
            net_real += real_buf[k];  // real part encodes net direction in each band
        }

        // Sign: +1 if net real component is positive (net buy at this frequency), -1 sell
        int8_t sign = (net_real > 1e-9) ? 1 : (net_real < -1e-9 ? -1 : 0);

        double f_center      = BAND_CENTERS[b];
        double weighted_force = (f_center > 0.0) ? (1.0 / f_center) * energy * sign : 0.0;

        fv.bands[b].band          = static_cast<FrequencyBand>(b);
        fv.bands[b].energy        = energy;
        fv.bands[b].sign          = sign;
        fv.bands[b].weighted_force = weighted_force;

        fv.total_force += weighted_force;
        total_energy   += energy;
    }

    // institutional_score = (INSTITUTIONAL + FUND) / total (matches institutional_score.py)
    double low_freq_energy = fv.bands[0].energy + fv.bands[1].energy;
    fv.institutional_score = (total_energy > 0.0) ? low_freq_energy / total_energy : 0.0;

    // dominant_band: band with highest energy
    int dominant = 0;
    for (int b = 1; b < 4; ++b) {
        if (fv.bands[b].energy > fv.bands[dominant].energy) dominant = b;
    }
    fv.dominant_band = static_cast<FrequencyBand>(dominant);

    return fv;
}

} // namespace p6
