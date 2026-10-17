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

The spectral dye density curves in scripts/print_dyes/kodak_ra4_e4021.csv (Kodak
E-4021, 2009) and scripts/print_dyes/kodak_2393_h1_2393.csv (Kodak H-1-2393t,
1998), and the 2393 characteristic curves that the Kodak Vision Premier 2393
profile in negpy/features/exposure/papers.py is fitted to, are derived from
spektrafilm by Andrea Volpato, licensed CC BY-SA 4.0:

  https://github.com/andreavolpato/spektrafilm
  https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt

Modified by NegPy: the CSVs keep each profile's channel_density on its 5 nm grid
where all three dyes are defined; print_luts/kodak_e4021.npz and kodak_2393.npz
are baked from them, and papers.py holds the tone constants fitted to that data. NegPy distributes them under the GPL v3, which
Creative Commons lists as a BY-SA compatible license.

The print LUTs negpy/features/exposure/print_luts/fuji_ca_dpii.npz,
fuji_ca_hd.npz and fuji_ca_pearl.npz, and the paper white, base tint and Dmax
of the Fujicolor Crystal Archive stocks in papers.py, are derived from
Fujifilm's Frontier-PD printer profiles for Crystal Archive DPII, HD and Pearl
paper (FUJIFILM, made with Argyll CMS). The profiles are not distributed with
NegPy.

scripts/crosstalk_dyes/kodak_vision3_dyes.csv is the mean of spektrafilm's
traces of the Kodak VISION3 50D, 200T, 250D and 500T dye curves (H-1-5203,
H-1-5207, H-1-5213, H-1-5219), resampled to 5 nm, under the same license. The
Kodak consumer and VERITA 200D crosstalk estimates in crosstalk/ are fitted with
it.
