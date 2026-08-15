The IR dust reconstruction in negpy/features/retouch/logic.py ports algorithm
concepts (continuous defect score, score-weighted multiscale reconstruction,
original-floor writer rule, interior-radius routing with feathered compositing)
from digital-fauxice, Copyright (c) 2026 Rohan Pandula, MIT License.
https://github.com/rohanpandula/digital-fauxice

The OpenICE IR reconstruction method in negpy/features/retouch/openice.py ports
the algorithm from openICE, Copyright (C) 2026 <a6o>, GNU General Public License
v3.0 — the same licence NegPy is released under.
https://github.com/marcinz606/openICE

openICE is an independent reimplementation of Applied Science Fiction's Digital
ICE, for interoperability and research. Digital ICE is a trademark of Eastman
Kodak Company; Nikon, Coolscan and Nikon Scan are trademarks of Nikon
Corporation. Neither openICE nor NegPy is affiliated with or endorsed by them;
the names identify only the formats and hardware being interoperated with.

The Plustek USB driver lives in the separate [pyopticfilm](https://github.com/jboneng/pyopticfilm) package. NegPy integrates it via `negpy/infrastructure/scanners/plustek_backend.py`. That driver includes
material derived from or informed by the SANE Project genesys backend
(Scanner Access Now Easy), GNU GPL:

  https://gitlab.com/sane-project/backends

Relevant upstream areas include (non-exhaustive): backend/genesys USB protocol,
register tables, motor/sensor/frontend tables, and image pipeline helpers.
Plustek, OpticFilm, and related names are trademarks of their respective owners;
NegPy is not affiliated with or endorsed by Plustek Inc.
