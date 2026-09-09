"""P2.13 — dekorelacja osi depth↔T/mazur."""

from __future__ import annotations

from recall_data import axis_correlation_report, generate_recall_batch


def main():
    legacy = generate_recall_batch(4, decorrelate_axes=False)
    deco = generate_recall_batch(4, decorrelate_axes=True)
    c_leg = axis_correlation_report(legacy)
    c_dec = axis_correlation_report(deco)
    print("legacy", c_leg)
    print("decorr", c_dec)
    # po dekorelacji |corr|(depth,T) powinno być wyraźnie niższe niż legacy
    assert c_dec["|corr|(depth,T)"] < c_leg["|corr|(depth,T)"] - 0.2
    assert c_dec["|corr|(depth,T)"] < 0.35
    print("OK — axes decorrelated")


if __name__ == "__main__":
    main()
