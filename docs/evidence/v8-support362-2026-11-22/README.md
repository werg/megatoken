# Prepared objective-step inventory correction — 22 November 2026

The source-identity correction ac2d898c03b0c1e6ecf2c24860b76523f2c3f6f8 passed 57 foundation controls with zero skips and clean Ruff in native run 45007426937. Producer 362c7d56cb4b26db01f0c0a43639894591a82c10 / ac2d898 again passed all 64 original-source/native/token/packing and strict 48/12/4 prepared-supply gates in run 45007442526. It then failed before optimizer work because an ordinary step may retain records with no objective loss plans, while the inventory zipped both arrays unconditionally.

Availability now counts every policy-masked physical record independently of objective plans. available.lossDenominators counts only existing plans, regardless of effective weights. Positive-weight exposure remains separately derived by the actual native collector. An empty plan list contributes no denominators; a nonempty misaligned list still refuses via strict pairing. Regression controls cover both distinctions.

The full Actions failure and the two exact underlying inventory-stage tracebacks are retained. The full 57-control foundation pass is also retained. Actual optimizer coverage remains unqualified until the corrected full ordinary pass and exact checkpoint census succeed; no source, occurrence, token-position or learned-quality denominator is inflated.
