"""Vị trí hộp theo seed: phủ ĐỀU vùng gắp được (hình quạt quanh trục khớp 1) thay cho ô nhỏ ±3 cm.

Cách làm (xác định theo seed, không phụ thuộc số ngẫu nhiên của môi trường):
    - Hình quạt r in [r_min, r_max], góc in [angle_min, angle_max] (config.PICK_REGION) chia thành na x nr ô, các vòng bán kính chia
      đều DIỆN TÍCH nên mỗi ô có diện tích bằng nhau.
    - Seed k dùng ô thứ perm[k % số ô] (hoán vị cố định, để seed liền kề không nằm cạnh nhau), điểm rơi ngẫu nhiên (theo seed) trong ô.
    - Ô trùng sát bát hoặc sát mép bàn thì rút lại trong chính ô đó; hết cách thì rút trong cả hình quạt.
    - Seed trong config.BOX_FIXED_SEEDS (1-6) dùng đúng vị trí đã đặt (các trường hợp cực biên).
Mọi seed 0-199 phủ trọn 200 ô (trừ các ô bị bát/mép bàn loại, được thay bằng điểm khác trong quạt).
"""

import math

import numpy as np

from . import config

_PERM = None


def _perm(n):
    global _PERM
    if _PERM is None or len(_PERM) != n:
        _PERM = np.random.default_rng(20240607).permutation(n)
    return _PERM


def _valid(xy):
    """Hộp phải nằm trong mặt bàn, cách bát đủ xa."""
    if abs(xy[0]) > config.TABLE_HALF[0] - 0.03 or abs(xy[1]) > config.TABLE_HALF[1] - 0.03:
        return False
    return math.hypot(xy[0] - config.BOWL_DEFAULT_POS[0], xy[1] - config.BOWL_DEFAULT_POS[1]) >= config.BOX_BOWL_CLEARANCE


def _polar(r, a_deg):
    th = math.radians(a_deg)
    return (config.PICK_PAN_XY[0] + r * math.sin(th), config.PICK_PAN_XY[1] + r * math.cos(th))


def cell_bounds(cell):
    """(r0, r1, a0, a1) của ô số `cell`; chỉ số ô = ir * na + ia."""
    R, na_nr = config.PICK_REGION, config.BOX_SECTOR_CELLS
    na, nr = na_nr
    ia, ir = cell % na, cell // na
    a_lo, a_hi = R["angle_min_deg"], R["angle_max_deg"]
    a0 = a_lo + (a_hi - a_lo) * ia / na
    a1 = a_lo + (a_hi - a_lo) * (ia + 1) / na
    area0, area1 = R["r_min"] ** 2, R["r_max"] ** 2          # chia đều theo r^2 = chia đều diện tích
    r0 = math.sqrt(area0 + (area1 - area0) * ir / nr)
    r1 = math.sqrt(area0 + (area1 - area0) * (ir + 1) / nr)
    return r0, r1, a0, a1


def box_xy_for_seed(seed):
    """Vị trí (x, y) của hộp cho `seed`. seed None -> ô ngẫu nhiên."""
    if seed is not None and int(seed) in config.BOX_FIXED_SEEDS:
        return tuple(config.BOX_FIXED_SEEDS[int(seed)][1])
    na, nr = config.BOX_SECTOR_CELLS
    n = na * nr
    rng = np.random.default_rng(None if seed is None else (int(seed) * 7919 + 13))
    cell = int(_perm(n)[int(seed) % n]) if seed is not None else int(rng.integers(n))
    r0, r1, a0, a1 = cell_bounds(cell)
    for _ in range(40):                                        # trong chính ô
        r = math.sqrt(rng.uniform(r0 * r0, r1 * r1))
        xy = _polar(r, rng.uniform(a0, a1))
        if _valid(xy):
            return xy
    R = config.PICK_REGION
    for _ in range(200):                                       # cả hình quạt
        r = math.sqrt(rng.uniform(R["r_min"] ** 2, R["r_max"] ** 2))
        xy = _polar(r, rng.uniform(R["angle_min_deg"], R["angle_max_deg"]))
        if _valid(xy):
            return xy
    return _polar(0.5 * (R["r_min"] + R["r_max"]), 0.0)


def fixed_seed_name(seed):
    """Tên trường hợp cực biên của seed 1-6, hoặc None."""
    v = config.BOX_FIXED_SEEDS.get(int(seed))
    return v[0] if v else None


def is_unreachable(xy):
    """Check the conservative stop limits in the tabletop workspace."""
    dx, dy = xy[0] - config.PICK_PAN_XY[0], xy[1] - config.PICK_PAN_XY[1]
    ang = math.degrees(math.atan2(dx, dy))
    return (
        abs(ang) > config.PAN_LIMIT_DEG + config.PAN_UNREACHABLE_MARGIN_DEG
        or math.hypot(dx, dy) > config.PICK_UNREACHABLE_RADIUS
    )
