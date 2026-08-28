"""Section 3 of emg_proxy.py: the PSD diagnostic at the temporal ring, and the
pre-declared informativeness call on R2. Split out 2026-08-26; the body is
verbatim from that file."""

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

import numpy as np

from common import TEMPORAL
from emg_setup import hr, sub


def run_psd(D):
    raw_base, SFREQ = D.raw_base, D.SFREQ

    # =============================================================================
    # 3. PSD diagnostic
    # =============================================================================
    hr("3. PSD AT THE TEMPORAL RING, ON THE UNFILTERED AVERAGE-REFERENCED RECORDING")

    print("\nRead this table for two things, both of which are falsifiers:")
    print("  (i)  a NARROW SPIKE dominating the probe band means the feature is line")
    print("       residual or an alias, not band power. In particular a 120 Hz line")
    print("       harmonic, if the amplifier's anti-alias filter let it through before")
    print("       160 Hz sampling, ALIASES TO 40 Hz, the probe's lower passband edge.")
    print("       Aliased content cannot be notched out after the fact, so this is")
    print("       checked by inspection here and by nothing else.")
    print("  (ii) if the 65-75 Hz median is below 0.10x the 40-55 Hz median, the")
    print("       amplifier has rolled the band off, R2 sits at the noise floor, and")
    print("       R2 is DECLARED UNINFORMATIVE and not used as an arbiter. The 0.10")
    print("       figure is a judgement call fixed in advance, and is stated as such.")

    psd = raw_base.copy().pick(TEMPORAL).compute_psd(
        method="welch", fmin=25.0, fmax=79.0, n_fft=int(SFREQ * 2), n_overlap=int(SFREQ),
        verbose="error",
    )
    psd_data, freqs = psd.get_data(return_freqs=True)
    psd_med = np.median(psd_data, axis=0)  # median across the 8 temporal channels

    sub("Median PSD across the 8 temporal channels, 25 to 79 Hz")
    scale = psd_med.max()
    for f, v in zip(freqs, psd_med):
        bar = "#" * int(round(56 * v / scale))
        mark = ""
        if abs(f - 60.0) < 0.6:
            mark = "  <-- 60 Hz line"
        elif abs(f - 40.0) < 0.6:
            mark = "  <-- 40 Hz, probe lower edge / 120 Hz alias lands here"
        print(f"  {f:5.1f} Hz {v:10.3e} |{bar:<56}|{mark}")

    def prominence(f0):
        """How far a bin stands above its own neighbourhood, excluding the bin's
    immediate skirts. Printed so that falsifier 5, which the pre-registration
    leaves to visual inspection, has a reproducible number attached to it. This
    is a DESCRIPTIVE aid to that inspection, not a new decision rule invented
    after the fact."""
        near = (np.abs(freqs - f0) >= 2.0) & (np.abs(freqs - f0) <= 5.0)
        at = np.argmin(np.abs(freqs - f0))
        return psd_med[at] / np.median(psd_med[near])


    sub("Spike prominence, for the falsifier-5 inspection")
    print(f"  60.0 Hz bin / its 2-5 Hz neighbourhood : {prominence(60.0):.2f}x   "
          f"the mains line, notched out of the primary band")
    print(f"  40.0 Hz bin / its 2-5 Hz neighbourhood : {prominence(40.0):.2f}x   "
          f"where a 120 Hz harmonic would alias to")
    print("  A band DOMINATED by a narrow spike would fail falsifier 5. A modest")
    print("  local maximum at 40 Hz is reported as what it is and is not explained")
    print("  away: it is consistent with a small aliased 120 Hz harmonic, and 160 Hz")
    print("  sampling makes that indistinguishable from genuine 40 Hz content after")
    print("  the fact. Whatever it is, it is present in BOTH the real-data row and")
    print("  every ladder rung, so the ladder's sensitivity threshold is measured in")
    print("  situ and already carries it.")

    PSD_BANDS = {"40-55": (40.0, 55.0), "56-64": (56.0, 64.0), "65-75": (65.0, 75.0)}
    psd_band_med = {}
    for name, (lo, hi) in PSD_BANDS.items():
        m = (freqs >= lo) & (freqs <= hi)
        psd_band_med[name] = float(np.median(psd_data[:, m]))

    sub("Median PSD by band (V^2/Hz), across all 8 temporal channels")
    for name, v in psd_band_med.items():
        print(f"  {name:>6} Hz : {v:.4e}")
    r2_ratio = psd_band_med["65-75"] / psd_band_med["40-55"]
    print(f"\n  65-75 / 40-55 ratio = {r2_ratio:.3f}  (threshold 0.100, fixed in advance)")
    R2_INFORMATIVE = r2_ratio >= 0.10
    if R2_INFORMATIVE:
        print("  R2 IS INFORMATIVE and may be used as an arbiter.")
    else:
        print("  R2 IS DECLARED UNINFORMATIVE, as pre-registered. Its band sits at the")
        print("  noise floor, so a null there confirms nothing and is not used as an")
        print("  arbiter for anything below.")
    line_ratio = psd_band_med["56-64"] / psd_band_med["40-55"]
    print(f"  56-64 / 40-55 ratio = {line_ratio:.3f}  (how much the 60 Hz line dominates,")
    print(f"  which is why the primary band notches it out rather than straddling it.)")

    return R2_INFORMATIVE
