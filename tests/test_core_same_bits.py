import hashlib

import negpy_core
import numpy as np
import pytest

from negpy.features.exposure.logic import apply_characteristic_curve
from negpy.features.toning.logic import apply_chemical_toning
from negpy.features.transparency.logic import apply_transfer_curve
from negpy.kernel.image.logic import WORKING_LAB

# Recorded on Linux x86-64, where the cv2 oracles pass. Every release platform must give these
# bytes; CI runs this file on each one. A port adds its case here; a hash only changes with a
# decided change of output.
_RNG = np.random.default_rng(2026)
_IMG3 = _RNG.random((97, 131, 3), dtype=np.float32)
_IMG1 = (_RNG.random((83, 61), dtype=np.float32) - 0.4) * 5
_GRAY = negpy_core.bgr_to_gray((_IMG3 * 255).astype(np.uint8))
_M = np.array([[0.58, 0.19, 0.19], [0.30, 0.63, 0.08], [0.03, 0.07, 0.99]], np.float32)


_LUT = np.ascontiguousarray(np.clip(np.cumsum(_RNG.random((9, 9, 9, 3), dtype=np.float32), axis=0) / 6, 0, 1))
# Plain arithmetic only: NumPy's own float32 power differs between CPUs.
_LIN = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
_RAMP = np.ascontiguousarray(np.tile(_LIN * (np.float32(2.0) - _LIN), (3, 1)))


def _filled(fill):
    out = np.empty_like(_IMG3)
    fill(out)
    return out


_LAB = (_IMG3 * np.array([100.0, 160.0, 160.0], np.float32) - np.array([0.0, 80.0, 80.0], np.float32)).astype(np.float32)

_CASES = {
    "percentiles": lambda: np.array(negpy_core.percentiles(_IMG1.ravel(), [0.5, 1.0, 37.0, 50.0, 99.0]), np.float32),
    "resize_area_general": lambda: negpy_core.resize_area(_IMG3, (47, 31)),
    "resize_linear_up": lambda: negpy_core.resize_linear(_IMG3, (300, 177)),
    "resize_area_up": lambda: negpy_core.resize_area(_IMG1, (100, 120)),
    "resize_nearest": lambda: negpy_core.resize_nearest(_GRAY, (300, 41)),
    "resize_area_2x2": lambda: negpy_core.resize_area(_IMG1[:82, :60].copy(), (30, 41)),
    "gaussian_5x5": lambda: negpy_core.gaussian_blur(_IMG3, (5, 5), 0.0),
    "gaussian_sigma2_replicate": lambda: negpy_core.gaussian_blur(_IMG1, (0, 0), 2.0, 0.0, 1),
    "gaussian_kernel_17": lambda: np.array(negpy_core.gaussian_kernel(17, 2.0), np.float32),
    "sobel_x": lambda: negpy_core.sep_filter2d(_IMG1, np.array([-1, 0, 1], np.float32), np.array([1, 2, 1], np.float32)),
    "bgr_to_gray": lambda: _GRAY,
    "gaussian_blur_u8": lambda: negpy_core.gaussian_blur_u8(_GRAY, (5, 5), 0.0),
    "otsu": lambda: negpy_core.threshold_otsu(_GRAY)[1],
    "canny": lambda: negpy_core.canny(_GRAY, 40, 160),
    "find_external_contours": lambda: np.concatenate(
        [c.ravel() for c in negpy_core.find_external_contours((_IMG1 > 0.5).astype(np.uint8) * np.uint8(255))[0]]
    ),
    "morph_close_u8": lambda: negpy_core.morphology_ex((_IMG1 > 0.5).astype(np.uint8) * 255, 3, np.ones((9, 7), np.uint8), 2),
    "to_log_density": lambda: negpy_core.to_log_density(np.ascontiguousarray(_IMG1 * _IMG1)),
    "oetf_encode": lambda: negpy_core.oetf_encode(_IMG3, np.float32(256.0 / 563.0)),
    "rgb_to_lab": lambda: negpy_core.rgb_to_lab(
        _IMG3, _M, np.array([0.95047, 1.0, 1.08883], np.float32), np.float32(0.008856), np.float32(7.787)
    ),
    "lab_to_rgb": lambda: negpy_core.lab_to_rgb(
        _LAB, _M.T.copy(), np.array([0.95047, 1.0, 1.08883], np.float32), np.float32(0.008856), np.float32(7.787)
    ),
    "skin_chroma_rein": lambda: negpy_core.skin_chroma_rein(_LAB, np.float32(0.6)),
    "matmul_3x3": lambda: negpy_core.matmul_3x3(_IMG3, _M),
    "gamut_aware_chroma_scale": lambda: negpy_core.gamut_aware_chroma_scale(
        _LAB, np.float32(1.8), _M.T.copy(), np.array([0.95047, 1.0, 1.08883], np.float32), np.float32(0.008856), np.float32(7.787), 10
    ),
    "print_curve": lambda: apply_characteristic_curve(
        _IMG3 * 0.8 - 0.1, (0.45, 2.1), (0.5, 2.0), (0.55, 1.9), toe=0.3, shoulder=0.2, midtone_gamma=0.2, shadow_density=-0.1, bpc=True
    ),
    "chemical_toning": lambda: apply_chemical_toning(_IMG3, 0.3, 0.4, 0.3, 0.2, 0.3, 0.4),
    "sensor_log_unmix": lambda: _filled(
        lambda out: negpy_core.sensor_log_unmix(_IMG3, _M, np.array([0.01, -0.02, 0.03], np.float32), np.full(3, 1e-3, np.float32), out)
    ),
    "lut_f32": lambda: negpy_core.apply_lut_f32(_IMG3, _LUT),
    "matrix_trc_u8": lambda: negpy_core.matrix_trc_u8((_IMG3 * 255).astype(np.uint8), np.ascontiguousarray(_RAMP[:, ::256]), _M, _RAMP),
    "vignette": lambda: negpy_core.apply_vignette(_IMG3, 0.7, 0.6, 0.3),
    "flat_curve": lambda: negpy_core.flat_curve(_IMG3, 0.65, 0.1, (0.013, -0.02, 0.0), _M),
    "transfer_curve": lambda: apply_transfer_curve(
        _IMG3,
        0.1,
        1.1,
        (0.2, 0.3, 0.1),
        (0.2, 0.0, 0.25),
        (0.01, 0.0, -0.02),
        shadow_density=0.15,
        shadow_cmy=(0.02, 0.0, 0.0),
        cast_gain=(1.02, 1.0, 0.97),
        separation=1.3,
        damping=0.5,
    ),
    "matrix_trc_u16": lambda: negpy_core.matrix_trc_u16(
        (_IMG3 * 65535).astype(np.uint16), np.ascontiguousarray(np.tile(_LIN, (3, 1))), _M, np.ascontiguousarray(_RAMP[:, ::16])
    ),
    "usm_sharpen": lambda: negpy_core.usm_sharpen(
        _IMG3, np.float32(2.0), 0.4, np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], np.float32), *WORKING_LAB
    ),
    "sharpen_grain": lambda: np.array(
        [negpy_core.sharpen_grain(negpy_core.gaussian_blur(_IMG3, (0, 0), 2.0) * np.float32(0.4), (0.2973769, 0.6273491, 0.0752741))],
        np.float32,
    ),
    "sabattier": lambda: negpy_core.sabattier(
        _IMG3,
        (np.float32(0.9), np.float32(0.08), np.float32(0.1), np.float32(1.3), np.float32(2.0)),
        np.array([0.25, 0.5, 0.25], np.float32),
    ),
    "lith": lambda: negpy_core.lith(
        _IMG3,
        [np.float32(v) for v in (2.0, 0.3, 0.7, 0.6, 0.6, 1.7, 0.14)],
        [np.float32(v) for v in (0.1, 0.35, 0.65, 1.0)],
        [(np.float32(a), np.float32(b)) for a, b in ((4.0, 12.0), (6.0, 18.0), (2.0, 8.0), (0.0, 0.5))],
        WORKING_LAB,
    ),
    "carrier": lambda: negpy_core.apply_carrier(
        _IMG3,
        9.0,
        0.7,
        0.5,
        0.4,
        (np.float32(0.98), np.float32(0.97), np.float32(0.95)),
        np.ascontiguousarray(np.repeat(_LIN[::1040][:64, None], 3, axis=1)),
        np.ascontiguousarray(np.tile(_LIN[::32] * np.float32(2.0) - np.float32(1.0), (8, 1))),
    ),
    "warp_affine": lambda: negpy_core.warp_affine(_IMG3, np.array([[0.998, 0.061, -3.1], [-0.061, 0.998, 4.2]]), (131, 97)),
    "warp_perspective": lambda: negpy_core.warp_perspective(
        _IMG1, np.array([[1.02, 0.03, -1.5], [-0.01, 0.97, 2.25], [1e-4, -2e-4, 1.0]]), (61, 83)
    ),
    "warp_affine_cubic_constant": lambda: negpy_core.warp_affine(
        _IMG3, np.array([[0.998, 0.061, -3.1], [-0.061, 0.998, 4.2]]), (140, 90), interpolation="cubic", border="constant"
    ),
    "remap_nearest": lambda: negpy_core.remap(
        _IMG1, *np.meshgrid(np.arange(70, dtype=np.float32) * 0.9 - 2, np.arange(90, dtype=np.float32) * 0.95 - 1), "nearest", "constant"
    ),
    "remap_radial": lambda: negpy_core.remap_radial(_IMG3, (65.0, 48.0), 0.93, -0.05, 1.0 / 6600.0),
}

_HASHES = {
    "percentiles": "0874152b6e9f91bd3a3b73e15a9d083d15b6e8c7c6f6a9279d9241df10a88464",
    "resize_area_general": "dbe0fb9e51b0e8bee54cc2bb35756bbf0934077b458c1a44ea27c2e5dd96a16f",
    "resize_linear_up": "9e6c1200f4f4f05bf46bbba858c67501044e514e47ce1afb50ac5a0e3e7cabee",
    "resize_area_up": "1b0aa67c89167cb289f780e36bdd5465c5d306200b3f1413510b6cfcd608fcac",
    "resize_nearest": "1b335732ac03cc2345941fba19ec591c930b03fd5abdfac82bdf5111984d9956",
    "resize_area_2x2": "cae2830d9848ee6f270000d57b3dc448138cc99cb5a2303108ac3e095b060488",
    "gaussian_5x5": "988a867049827fd5912f1950b8574b637d671c3565e0b7af87def261c9f0b084",
    "gaussian_sigma2_replicate": "2b9be6f95b7f66a03493795d8c7a5e1bb2565a28087578f1981a9cd329c8a74f",
    "gaussian_kernel_17": "5c2eb92d43931221106d80ca89a0b6dea1dca09b261b0fbf6ff51fdcae748d8b",
    "sobel_x": "ba798e861db64c102fee6b08fd7aa34d488cce431ae0d95fc3e7836e1dedb44a",
    "bgr_to_gray": "ea18764e8727ce823408e9962b86969c2de631c5f0246b1d33c3bc833101dac9",
    "gaussian_blur_u8": "7817f7444ec9c843b3c529098d650d653dd0081250770eb96bb378da20de046e",
    "otsu": "cff78130e93898fd1d8ff6080127920c71a5fa1bbecd734a35631b253866ffdb",
    "canny": "1e3125ae47a5dd1719f97531ffebffc79a6461ba1fae024d26d4d7a85b89b968",
    "find_external_contours": "ebb4302dd2bc7226a7ed24fbb5e16e14f407d5693ce47128a649cee9e4a098be",
    "morph_close_u8": "4be9a2976881ff83eb0b92a8ec484e048c61612324afbde73b6e0c89aa92865e",
    "to_log_density": "cc4d2086edf051826619f024a4f5d28c80d28528291d9df4610c1d3854144038",
    "oetf_encode": "44508d28673c678fc0ec7b5262a06d0053ea6b4bed4bead7bc49556bc60eb070",
    "rgb_to_lab": "02c583bf1f5fe9723e2cb0a3e1f34759d6b0778d3644e8e2cff2251e756b68f2",
    "lab_to_rgb": "34016e7f9aebcfdf780b67b96b128041c6ba1ac2a427f4e87b5413dcc7345eef",
    "skin_chroma_rein": "6def89b9f456f16abf406c351745bbb791f15bc69e5b03f2e478ed7391839b01",
    "matmul_3x3": "d80c524d23c1bc6b7d9592b638a5fb84d4b51691efac9b71b40fbf529e67c019",
    "gamut_aware_chroma_scale": "ef1a9474817c0bca19e3c847a4667ba47ba75829722a1c7939a77e135d8b7da4",
    "print_curve": "c06516de4316987b8cb37dcf977915373302d36b0bd6c1018e3e4bfba0929b09",
    "chemical_toning": "7ddd3949c04bd5af38177bfcd6576888092bf73ad173206c34b153d6cce5d648",
    "sensor_log_unmix": "f7eb8efe1bbcfe884fd2e849e99d0546fac2f4829d3fb558d54a0128e681e53b",
    "lut_f32": "467bb9fd495d4797d912f6aeabd34f9a61b56491105464a0c852ed524488b1c1",
    "matrix_trc_u8": "acec1eadffb13dfef8473a02fb250b4ad8f57ac211a6265661b77d3cad49ab16",
    "matrix_trc_u16": "8fd1af40718fc9bc59541e1128e46cb32e08ea05626f7cbd4cebe7905cecd02a",
    "vignette": "dea377797d6b0852a8dec1665c95aa87c9ab2b6a8e03f808e27140bc4b6566c6",
    "flat_curve": "e47f6084a58e19119c6e3364ebf017dfc7f039ec6c73e0eea8e942519021bf0b",
    "transfer_curve": "afdb5891162dca5abe4530bd098469b26a0be16a41df32f4f2928ddb8d37b943",
    "usm_sharpen": "7267d2a3976c9381eadde01fe11ea60d46345b4747283e6ac68a2aa4a025d0a8",
    "sharpen_grain": "4dcdc651f4aa9dca4a8678ad0e3db727332b8fa7b4ede77889b88d506df6e506",
    "sabattier": "f8956f67786c7c1d0e628af4b82fb12b6dde42e70da2504585f5bf20e7fbb461",
    "lith": "98dc05c62e191da2ad00c1f2327c48beff5af7e882be64cac5302cb0ea5099e5",
    "carrier": "97c9d045201ba345010f91b09ea07699fecc4e009d7738dc8591295b59e6081c",
    "warp_affine": "83562b3183484dcbf80fc65c4fb2f241d3b13ec9793cf43fd22bf11ea44d9e1d",
    "warp_perspective": "61e89302e793e24f1ad131b661e95186d7d9a82e8331a96164e338bec3ec1420",
    "warp_affine_cubic_constant": "a3d3d1a3d6c5e0e77686cfd37850199aa880943936e83edf8ada48307501cd69",
    "remap_nearest": "16ebb2a1c29640f25e4a490f6edd8817d8dc1f652b6aa001f72aa2ca34d33ac0",
    "remap_radial": "a23f3b21f15cb5d9db26cc532ec920dc8b194d73515a786d9febc93ae17370e0",
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_same_bits_on_every_platform(name):
    out = np.ascontiguousarray(_CASES[name]())
    assert hashlib.sha256(out.tobytes()).hexdigest() == _HASHES[name]


def test_every_case_has_a_hash():
    assert _CASES.keys() == _HASHES.keys()
