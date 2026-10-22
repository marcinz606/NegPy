The IR dust reconstruction in negpy/features/retouch/logic.py ports algorithm
concepts (continuous defect score, score-weighted multiscale reconstruction,
original-floor writer rule, interior-radius routing with feathered compositing)
from digital-fauxice, Copyright (c) 2026 Rohan Pandula, MIT License.
https://github.com/rohanpandula/digital-fauxice

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

The print LUTs negpy/features/exposure/print_luts/fuji_ca_dpii.npz and
fuji_ca_hd.npz, and the paper white, base tint and Dmax of the Fujicolor
Crystal Archive stocks in papers.py, are derived from Fujifilm's Frontier-PD
printer profiles for Crystal Archive DPII and HD paper (FUJIFILM, made with Argyll CMS). The profiles are not distributed with
NegPy.

The print LUT negpy/features/exposure/print_luts/frontier.npz is derived from the
color reproduction tables (ColorData.dat) of the Fujifilm Frontier
350/370 A1 system software v7.6. The tables are not distributed with NegPy.

scripts/crosstalk_dyes/kodak_vision3_dyes.csv is the mean of spektrafilm's
traces of the Kodak VISION3 50D, 200T, 250D and 500T dye curves (H-1-5203,
H-1-5207, H-1-5213, H-1-5219), resampled to 5 nm, under the same license. The
Kodak consumer and VERITA 200D crosstalk estimates in crosstalk/ are fitted with
it.

The DX numbers, the Kodak Ultra Max edge code and two respool sources in the
`edge` entries of gear/film_stocks.json come from The Big Film Database
(dxdatabase) © 2018 by Dxdatabase, licensed CC BY-SA 4.0 and provided as is,
without warranties:

  http://industrieplus.alwaysdata.net/dxdatabase/
  https://github.com/dxdatabase/Open-source-film-database
  https://thebigfilmdatabase.merinorus.com/
  https://creativecommons.org/licenses/by-sa/4.0/

Modified by NegPy: each four-digit DX number is split into the part 1 and part 2
of the film edge barcode. NegPy distributes them under the GPL v3, which
Creative Commons lists as a BY-SA compatible license.
