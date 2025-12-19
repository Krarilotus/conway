# conway.py
import sys
import argparse
import numpy as np
import pygame
import time
import random
import os
import math
import threading
from collections import deque

# =============================================================================
# BUILD FLAGS
# =============================================================================
BUILD_DEBUG = False  # set True before compiling your debug build

# =============================================================================
# OPTIONAL DEPENDENCIES (debug HUD)
# =============================================================================
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# =============================================================================
# OPTIONAL SCIPY (only needed for "largerthanlife")
# =============================================================================
try:
    import scipy.signal
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# =============================================================================
# Prevent display sleep while running (Windows)
# =============================================================================
if os.name == "nt":
    import ctypes
    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def keep_display_awake():
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
else:
    def keep_display_awake():
        pass

# =============================================================================
# Logging (important for --noconsole builds)
# =============================================================================
def log_to_temp(msg: str):
    if os.name != "nt":
        return
    try:
        p = os.path.join(os.environ.get("TEMP", "."), "conway_screensaver.log")
        with open(p, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass

# =============================================================================
# Multi-monitor helpers (Windows)
# =============================================================================
def get_monitor_rects_windows():
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG),
                    ("top", wintypes.LONG),
                    ("right", wintypes.LONG),
                    ("bottom", wintypes.LONG)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", wintypes.DWORD)]

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM
    )

    rects = []

    def _callback(hMonitor, hdc, lprcMonitor, dwData):
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi)):
            r = mi.rcMonitor
            rects.append((int(r.left), int(r.top), int(r.right), int(r.bottom)))
        return True

    cb = MonitorEnumProc(_callback)
    ok = user32.EnumDisplayMonitors(0, 0, cb, 0)
    return rects if ok else []

def bounding_box_from_rects(rects):
    left   = min(l for l, t, r, b in rects)
    top    = min(t for l, t, r, b in rects)
    right  = max(r for l, t, r, b in rects)
    bottom = max(b for l, t, r, b in rects)
    return left, top, right - left, bottom - top

def get_virtual_screen_bounds():
    if os.name == "nt":
        import ctypes
        user32 = ctypes.windll.user32
        SM_XVIRTUALSCREEN  = 76
        SM_YVIRTUALSCREEN  = 77
        SM_CXVIRTUALSCREEN = 78
        SM_CYVIRTUALSCREEN = 79
        vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        if vw > 0 and vh > 0:
            return int(vx), int(vy), int(vw), int(vh)

    pygame.display.init()
    info = pygame.display.Info()
    return 0, 0, int(info.current_w), int(info.current_h)

# =============================================================================
# CONFIG
# =============================================================================
FPS_MAX = 60
COLOR_BG = (10, 10, 15)

DEFAULT_COLORS = {
    0: (10, 10, 15),
    1: (0, 255, 128),
    2: (0, 120, 255),
    3: (255, 60, 60),
    4: (255, 220, 0),
}

VARIANTS = [
    'standard', 'highlife', 'daynight', 'seeds', 'immigration', 'quadlife',
    'generations', 'briansbrain', 'largerthanlife', 'margolus'
]

GRID_SIZE = 1  # computed in main()

def choose_grid_size_explosive(vw, vh):
    short = max(1, min(vw, vh))
    gs = int(round(short / 1000.0))
    return max(1, gs)

# =============================================================================
# Helpers: smooth transitions
# =============================================================================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

class Transition:
    def __init__(self, initial, lo=None, hi=None, min_dur=4.0, max_dur=10.0):
        self.current = float(initial)
        self.start = float(initial)
        self.target = float(initial)
        self.t0 = time.time()
        self.duration = 1.0
        self.next_change = self.t0 + random.uniform(min_dur, max_dur)
        self.lo = lo
        self.hi = hi
        self.min_dur = float(min_dur)
        self.max_dur = float(max_dur)

    def set_target(self, now, new_target, duration):
        if self.lo is not None and self.hi is not None:
            new_target = clamp(new_target, self.lo, self.hi)
        self.start = float(self.current)
        self.target = float(new_target)
        self.t0 = now
        self.duration = max(0.001, float(duration))
        self.next_change = now + self.duration

    def maybe_pick_new_target(self, now, pick_fn):
        if now >= self.next_change:
            dur = random.uniform(self.min_dur, self.max_dur)
            self.set_target(now, pick_fn(), dur)

    def update(self, now):
        t = (now - self.t0) / self.duration
        u = smoothstep(t)
        self.current = self.start + (self.target - self.start) * u
        if self.lo is not None and self.hi is not None:
            self.current = clamp(self.current, self.lo, self.hi)
        return float(self.current)

class ColorTransition:
    def __init__(self, initial_rgb, min_dur=4.0, max_dur=10.0):
        self.current = tuple(initial_rgb)
        self.start = tuple(initial_rgb)
        self.target = tuple(initial_rgb)
        now = time.time()
        self.t0 = now
        self.duration = 1.0
        self.next_change = now + random.uniform(min_dur, max_dur)
        self.min_dur = float(min_dur)
        self.max_dur = float(max_dur)

    def set_target(self, now, new_rgb, duration):
        self.start = self.current
        self.target = tuple(new_rgb)
        self.t0 = now
        self.duration = max(0.001, float(duration))
        self.next_change = now + self.duration

    def maybe_pick_new_target(self, now, pick_fn):
        if now >= self.next_change:
            dur = random.uniform(self.min_dur, self.max_dur)
            self.set_target(now, pick_fn(), dur)

    def update(self, now):
        t = (now - self.t0) / self.duration
        u = smoothstep(t)
        c = (
            int(self.start[0] + (self.target[0] - self.start[0]) * u),
            int(self.start[1] + (self.target[1] - self.start[1]) * u),
            int(self.start[2] + (self.target[2] - self.start[2]) * u),
        )
        self.current = c
        return c

# =============================================================================
# Disk sampling pool (fixed, no growth)
# =============================================================================
class RandomDiskPool:
    """
    Uniform disk sample: (cosθ * sqrt(u), sinθ * sqrt(u))
    """
    def __init__(self, size=1_200_000, seed=None):
        self.size = int(size)
        self.rng = np.random.default_rng(seed)

        theta = self.rng.random(self.size, dtype=np.float32) * (2.0 * math.pi)
        u = self.rng.random(self.size, dtype=np.float32)
        np.sqrt(u, out=u)

        self.cos = np.cos(theta, dtype=np.float32)
        self.sin = np.sin(theta, dtype=np.float32)
        self.su = u
        self.idx = 0

    def take(self, k: int):
        k = int(k)
        if k <= 0:
            return (0, 0, 0, 0)
        i = self.idx
        j = i + k
        if j <= self.size:
            self.idx = j if j < self.size else 0
            return (i, j, 0, 0)
        else:
            j1 = self.size
            k2 = j - self.size
            self.idx = k2
            return (i, j1, 0, k2)

def points_for_fill_fraction(radius_cells: int, fill_fraction: float) -> int:
    """
    Density semantics: expected unique coverage p per draw call.
      p ≈ 1 - exp(-k/A)  =>  k = -A ln(1-p), A = pi r^2
    """
    r = max(1, int(radius_cells))
    p = clamp(float(fill_fraction), 0.0, 0.999999)
    if p <= 0.0:
        return 0
    area = math.pi * float(r * r)
    k = int(-area * math.log(1.0 - p))
    return max(0, k)

# =============================================================================
# Paint batch (fixed buffers)
# =============================================================================
class PaintBatch:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.xs = np.empty(self.capacity, dtype=np.int32)
        self.ys = np.empty(self.capacity, dtype=np.int32)
        self.vals = np.empty(self.capacity, dtype=np.int8)
        self.n = 0

        self.fx = np.empty(self.capacity, dtype=np.float32)
        self.fy = np.empty(self.capacity, dtype=np.float32)

    def reset(self):
        self.n = 0

    def add_disk_points(self, pool: RandomDiskPool, cx: int, cy: int, r: int, k: int, val: int, gw: int, gh: int):
        if k <= 0:
            return

        while k > 0:
            space = self.capacity - self.n
            if space <= 0:
                return
            take = k if k < space else space

            i1, j1, _, k2 = pool.take(take)
            outx = self.fx[self.n:self.n + take]
            outy = self.fy[self.n:self.n + take]

            s1 = j1 - i1
            if s1 > 0:
                np.multiply(pool.su[i1:j1], float(r), out=outx[:s1])
                np.multiply(pool.cos[i1:j1], outx[:s1], out=outx[:s1])
                np.add(outx[:s1], float(cx), out=outx[:s1])

                np.multiply(pool.su[i1:j1], float(r), out=outy[:s1])
                np.multiply(pool.sin[i1:j1], outy[:s1], out=outy[:s1])
                np.add(outy[:s1], float(cy), out=outy[:s1])

            if k2 > 0:
                o2x = outx[s1:s1 + k2]
                o2y = outy[s1:s1 + k2]
                np.multiply(pool.su[0:k2], float(r), out=o2x)
                np.multiply(pool.cos[0:k2], o2x, out=o2x)
                np.add(o2x, float(cx), out=o2x)

                np.multiply(pool.su[0:k2], float(r), out=o2y)
                np.multiply(pool.sin[0:k2], o2y, out=o2y)
                np.add(o2y, float(cy), out=o2y)

            np.rint(outx, out=outx)
            np.rint(outy, out=outy)

            np.copyto(self.xs[self.n:self.n + take], outx, casting="unsafe")
            np.copyto(self.ys[self.n:self.n + take], outy, casting="unsafe")

            # wrap (fast)
            np.mod(self.xs[self.n:self.n + take], gw, out=self.xs[self.n:self.n + take])
            np.mod(self.ys[self.n:self.n + take], gh, out=self.ys[self.n:self.n + take])

            self.vals[self.n:self.n + take] = np.int8(val)
            self.n += take
            k -= take

    def flush_into(self, grid: np.ndarray):
        if self.n <= 0:
            return
        grid[self.ys[:self.n], self.xs[:self.n]] = self.vals[:self.n]
        self.n = 0

# =============================================================================
# Game engine
# =============================================================================
class GameEngine:
    def __init__(self, width, height, variant):
        self.width = width
        self.height = height
        self.variant = variant
        self.step_count = 0
        self.parity = 0

        self.ltl_range = 5
        self.ltl_birth = (34, 45)
        self.ltl_survive = (33, 57)

    def get_neighbors_roll(self, grid_bool_int8):
        g = grid_bool_int8
        N  = np.roll(g,  1, axis=0)
        S  = np.roll(g, -1, axis=0)
        E  = np.roll(g, -1, axis=1)
        W  = np.roll(g,  1, axis=1)
        NE = np.roll(N, -1, axis=1)
        NW = np.roll(N,  1, axis=1)
        SE = np.roll(S, -1, axis=1)
        SW = np.roll(S,  1, axis=1)
        return N + S + E + W + NE + NW + SE + SW

    def get_colored_neighbors_roll(self, grid, out_counts):
        for s in [1, 2, 3, 4]:
            g_s = (grid == s).astype(np.int8)
            N  = np.roll(g_s,  1, axis=0)
            S  = np.roll(g_s, -1, axis=0)
            E  = np.roll(g_s, -1, axis=1)
            W  = np.roll(g_s,  1, axis=1)
            NE = np.roll(N, -1, axis=1)
            NW = np.roll(N,  1, axis=1)
            SE = np.roll(S, -1, axis=1)
            SW = np.roll(S,  1, axis=1)
            out_counts[s][...] = N + S + E + W + NE + NW + SE + SW
        return out_counts

    def update_from_to(self, grid_in: np.ndarray, grid_out: np.ndarray, colored_counts=None):
        g = grid_in
        new_g = grid_out
        new_g.fill(0)

        if self.variant in ['standard', 'highlife', 'daynight', 'seeds', 'generations']:
            alive = (g > 0).astype(np.int8)
            n = self.get_neighbors_roll(alive)
            if self.variant == 'standard':
                new_g[...] = (((g == 1) & ((n == 2) | (n == 3))) | ((g == 0) & (n == 3))).astype(np.int8)
            elif self.variant == 'highlife':
                new_g[...] = (((g == 1) & ((n == 2) | (n == 3))) | ((g == 0) & ((n == 3) | (n == 6)))).astype(np.int8)
            elif self.variant == 'daynight':
                survive = np.isin(n, [3, 4, 6, 7, 8])
                birth = np.isin(n, [3, 6, 7, 8])
                new_g[...] = (((g == 1) & survive) | ((g == 0) & birth)).astype(np.int8)
            elif self.variant == 'seeds':
                new_g[...] = ((g == 0) & (n == 2)).astype(np.int8)
            elif self.variant == 'generations':
                MAX_DECAY = 4
                stays_alive = (g == 1) & ((n == 2) | (n == 3))
                birth = (g == 0) & (n == 3)
                new_g[stays_alive] = 1
                starts_decay = (g == 1) & ~stays_alive
                new_g[starts_decay] = 2
                mask_decay = (g > 1)
                new_g[mask_decay] = g[mask_decay] + 1
                new_g[new_g > MAX_DECAY] = 0
                new_g[birth] = 1

        elif self.variant in ['immigration', 'quadlife']:
            if colored_counts is None:
                colored_counts = {s: np.zeros_like(g, dtype=np.int16) for s in [1, 2, 3, 4]}
            n_counts = self.get_colored_neighbors_roll(g, colored_counts)
            total_n = n_counts[1] + n_counts[2] + n_counts[3] + n_counts[4]
            survive = (g > 0) & ((total_n == 2) | (total_n == 3))
            birth_mask = (g == 0) & (total_n == 3)

            if self.variant == 'immigration':
                birth_color = np.ones_like(g, dtype=np.int8)
                birth_color[n_counts[2] > n_counts[1]] = 2
                new_g[...] = np.where(survive, g, 0).astype(np.int8)
                new_g[birth_mask] = birth_color[birth_mask]
            else:
                birth_vals = np.zeros_like(g, dtype=np.int8)
                stacked = np.stack([n_counts[1], n_counts[2], n_counts[3], n_counts[4]])
                max_c = np.max(stacked, axis=0)
                for s in [1, 2, 3, 4]:
                    birth_vals[n_counts[s] == max_c] = s
                new_g[...] = np.where(survive, g, 0).astype(np.int8)
                new_g[birth_mask] = birth_vals[birth_mask]

        elif self.variant == 'briansbrain':
            g_firing = (g == 2).astype(np.int8)
            n = self.get_neighbors_roll(g_firing)
            mask0 = (g == 0)
            new_g[mask0] = np.where(n[mask0] == 2, 2, 0).astype(np.int8)
            new_g[g == 2] = 1

        elif self.variant == 'largerthanlife':
            if not HAS_SCIPY:
                alive = (g > 0).astype(np.int8)
                n = self.get_neighbors_roll(alive)
                new_g[...] = (((g == 1) & ((n == 2) | (n == 3))) | ((g == 0) & (n == 3))).astype(np.int8)
            else:
                RANGE = self.ltl_range
                sz = 2 * RANGE + 1
                k = np.ones((sz, sz), dtype=np.int8)
                k[RANGE, RANGE] = 0
                n = scipy.signal.convolve2d((g > 0).astype(np.int8), k, mode='same', boundary='wrap')
                birth = (g == 0) & (n >= self.ltl_birth[0]) & (n <= self.ltl_birth[1])
                survive = (g == 1) & (n >= self.ltl_survive[0]) & (n <= self.ltl_survive[1])
                new_g[...] = (birth | survive).astype(np.int8)

        elif self.variant == 'margolus':
            shift = self.parity
            w = np.roll(np.roll(g, -shift, axis=1), -shift, axis=0).copy()
            A, B = w[0::2, 0::2], w[0::2, 1::2]
            C, D = w[1::2, 0::2], w[1::2, 1::2]
            sum_block = A + B + C + D
            do_rotate = (sum_block == 1) | (sum_block == 3)
            new_A = np.where(do_rotate, C, A)
            new_B = np.where(do_rotate, A, B)
            new_D = np.where(do_rotate, B, D)
            new_C = np.where(do_rotate, D, C)
            w[0::2, 0::2], w[0::2, 1::2] = new_A, new_B
            w[1::2, 0::2], w[1::2, 1::2] = new_C, new_D
            new_g[...] = np.roll(np.roll(w, shift, axis=0), shift, axis=1).astype(np.int8)
            self.parity = 1 - self.parity

        self.step_count += 1

# =============================================================================
# Autopilot cursors (one draw call per tick)
# =============================================================================
class AutoCursor:
    def __init__(self, gw, gh, max_brush, start_xy=None):
        self.gw = gw
        self.gh = gh
        self.max_brush = max_brush

        sx = gw / 2.0
        sy = gh / 2.0
        if start_xy is not None:
            sx, sy = start_xy

        self.x = Transition(sx, lo=0.0, hi=max(0.0, gw - 1.0), min_dur=4.0, max_dur=10.0)
        self.y = Transition(sy, lo=0.0, hi=max(0.0, gh - 1.0), min_dur=4.0, max_dur=10.0)

        self.brush = Transition(
            initial=min(20.0, max_brush),
            lo=10.0, hi=float(max_brush),
            min_dur=4.0, max_dur=10.0
        )

        # IMPORTANT: per-tick fill is tiny (60Hz). These values WON'T freeze anything;
        # they only change visual density.
        self.fill = Transition(0.0060, lo=0.0010, hi=0.0200, min_dur=4.0, max_dur=10.0)
        self.layer = Transition(1.0, lo=0.5, hi=1.5, min_dur=4.0, max_dur=10.0)

    def pick_value(self, variant):
        if variant == 'immigration':
            return 1 if random.random() < 0.5 else 2
        if variant == 'quadlife':
            return random.randint(1, 4)
        if variant == 'briansbrain':
            return 2
        return 1

    def tick(self, now):
        self.x.maybe_pick_new_target(now, lambda: random.uniform(0.0, max(0.0, self.gw - 1.0)))
        self.y.maybe_pick_new_target(now, lambda: random.uniform(0.0, max(0.0, self.gh - 1.0)))
        self.brush.maybe_pick_new_target(now, lambda: random.uniform(10.0, float(self.max_brush)))
        self.fill.maybe_pick_new_target(now, lambda: random.uniform(self.fill.lo, self.fill.hi))
        self.layer.maybe_pick_new_target(now, lambda: random.uniform(self.layer.lo, self.layer.hi))

        x = self.x.update(now)
        y = self.y.update(now)
        b = self.brush.update(now)
        p = self.fill.update(now)
        layer = self.layer.update(now)

        r = max(1, int(round(b)) - 1)
        k = points_for_fill_fraction(r, p)
        k = int(k * layer)
        return int(round(x)), int(round(y)), r, max(0, k)

class AutoPilot:
    def __init__(self, gw, gh, monitor_rects_px, virtual_origin_px):
        self.gw = gw
        self.gh = gh

        # capped to 1/3 of prior 60 => 20
        self.speed_ips = Transition(12.0, lo=5.0, hi=20.0, min_dur=4.0, max_dur=10.0)

        self.colors = {
            0: ColorTransition(DEFAULT_COLORS[0], min_dur=8.0, max_dur=14.0),
            1: ColorTransition(DEFAULT_COLORS[1], min_dur=4.0, max_dur=10.0),
            2: ColorTransition(DEFAULT_COLORS[2], min_dur=4.0, max_dur=10.0),
            3: ColorTransition(DEFAULT_COLORS[3], min_dur=4.0, max_dur=10.0),
            4: ColorTransition(DEFAULT_COLORS[4], min_dur=4.0, max_dur=10.0),
        }

        now = time.time()
        self.next_variant = now + random.randint(60, 120)
        self.next_reset = now + random.randint(60, 120)

        self.max_brush = max(10, int(min(gw, gh) / 5))

        self.cursors = []
        vx, vy = virtual_origin_px

        if os.name == "nt" and monitor_rects_px:
            for (l, t, r, b) in monitor_rects_px:
                gl = int((l - vx) // GRID_SIZE)
                gt = int((t - vy) // GRID_SIZE)
                gr = int((r - vx) // GRID_SIZE)
                gb = int((b - vy) // GRID_SIZE)

                gl = int(clamp(gl, 0, gw - 1))
                gt = int(clamp(gt, 0, gh - 1))
                gr = int(clamp(gr, 1, gw))
                gb = int(clamp(gb, 1, gh))

                n = random.randint(1, 3)
                for _ in range(n):
                    sx = random.uniform(gl, max(gl + 1, gr) - 1)
                    sy = random.uniform(gt, max(gt + 1, gb) - 1)
                    self.cursors.append(AutoCursor(gw, gh, self.max_brush, start_xy=(sx, sy)))
        else:
            for _ in range(random.randint(1, 3)):
                self.cursors.append(AutoCursor(gw, gh, self.max_brush))

    def init_grid(self, engine: GameEngine, grid: np.ndarray):
        gw, gh = self.gw, self.gh
        if engine.variant == 'briansbrain':
            grid[...] = np.random.choice([0, 2], size=(gh, gw), p=[0.92, 0.08]).astype(np.int8)
        else:
            grid[...] = np.random.choice([0, 1], size=(gh, gw), p=[0.86, 0.14]).astype(np.int8)
            if engine.variant == 'immigration':
                grid[...] = np.random.choice([0, 1, 2], size=(gh, gw), p=[0.82, 0.09, 0.09]).astype(np.int8)

    def pick_color(self):
        return (random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))

    def tick(self, now):
        self.speed_ips.maybe_pick_new_target(now, lambda: random.uniform(5.0, 20.0))
        speed_ips = self.speed_ips.update(now)

        for k in [1, 2, 3, 4]:
            self.colors[k].maybe_pick_new_target(now, self.pick_color)
        self.colors[0].maybe_pick_new_target(now, lambda: DEFAULT_COLORS[0])
        color_map = {k: self.colors[k].update(now) for k in [0, 1, 2, 3, 4]}
        return speed_ips, color_map

# =============================================================================
# Shared state
# =============================================================================
class SharedState:
    def __init__(self, buffers):
        self.buffers = buffers
        self.lock = threading.Lock()
        self.stop = False

        self.pub_idx = 0
        self.render_idx = 0

        self.paint_lock = threading.Lock()
        self.paint_q = deque()

        self.metrics = {
            "speed_ips": 0.0,
            "colors": DEFAULT_COLORS.copy(),
            "variant": "standard",
            "step_count": 0,
            "sim_ips": 0.0,
            "error": "",
        }

# =============================================================================
# Sim thread worker (NEVER FREEZE VERSION)
# =============================================================================
class SimWorker(threading.Thread):
    def __init__(self, shared: SharedState, gw, gh, monitor_rects, virtual_origin, debug_enabled: bool):
        super().__init__(daemon=True)
        self.shared = shared
        self.gw = gw
        self.gh = gh
        self.debug_enabled = debug_enabled

        self.engine = GameEngine(gw, gh, variant='standard')
        self.autopilot = AutoPilot(gw, gh, monitor_rects, virtual_origin)

        self.disk_pool = RandomDiskPool(size=1_200_000)
        self.paint_batch = PaintBatch(capacity=1_500_000)

        self.colored_counts = {s: np.zeros((gh, gw), dtype=np.int16) for s in [1, 2, 3, 4]}

        self.cur_idx = 0
        self.autopilot.init_grid(self.engine, self.shared.buffers[self.cur_idx])
        with self.shared.lock:
            self.shared.pub_idx = self.cur_idx

        self.last_perf = time.perf_counter()
        self.sim_accum = 0.0

        self.logic_hz = 60.0
        self.logic_accum = 0.0

        self.sim_steps = 0
        self.sim_last = time.perf_counter()

    def _pick_free_idx(self, forbid_a: int, forbid_b: int):
        for i in range(len(self.shared.buffers)):
            if i != forbid_a and i != forbid_b:
                return i
        return (forbid_a + 1) % len(self.shared.buffers)

    def _drain_paint_commands(self):
        cmds = []
        with self.shared.paint_lock:
            while self.shared.paint_q:
                cmds.append(self.shared.paint_q.popleft())
        return cmds

    def _copy_on_write_if_needed(self, render_idx: int):
        """
        If our current buffer is being rendered, copy it to a free buffer before writing.
        This prevents:
          - frozen view (render stuck on a buffer we keep mutating but never publishing)
          - tearing / corruption
        """
        if self.cur_idx != render_idx:
            return
        new_idx = self._pick_free_idx(self.cur_idx, render_idx)
        self.shared.buffers[new_idx][...] = self.shared.buffers[self.cur_idx]
        self.cur_idx = new_idx

    def run(self):
        try:
            while True:
                with self.shared.lock:
                    if self.shared.stop:
                        return
                    render_idx = self.shared.render_idx

                nowp = time.perf_counter()
                dt = nowp - self.last_perf
                self.last_perf = nowp
                if dt < 0.0:
                    dt = 0.0
                if dt > 0.25:
                    dt = 0.25

                now = time.time()

                # update autopilot params
                speed_ips, colors = self.autopilot.tick(now)

                # reset/variant
                if now > self.autopilot.next_reset:
                    with self.shared.lock:
                        render_idx = self.shared.render_idx
                    self._copy_on_write_if_needed(render_idx)
                    self.autopilot.init_grid(self.engine, self.shared.buffers[self.cur_idx])
                    self.engine.step_count = 0
                    self.engine.parity = 0
                    self.autopilot.next_reset = now + random.randint(60, 120)
                    with self.shared.lock:
                        self.shared.pub_idx = self.cur_idx

                if now > self.autopilot.next_variant:
                    with self.shared.lock:
                        render_idx = self.shared.render_idx
                    self._copy_on_write_if_needed(render_idx)
                    self.engine.variant = random.choice(VARIANTS)
                    self.autopilot.init_grid(self.engine, self.shared.buffers[self.cur_idx])
                    self.engine.step_count = 0
                    self.engine.parity = 0
                    self.autopilot.next_variant = now + random.randint(60, 120)
                    for k in [1, 2, 3, 4]:
                        self.autopilot.colors[k].set_target(now, self.autopilot.pick_color(), random.uniform(4.0, 10.0))
                    with self.shared.lock:
                        self.shared.pub_idx = self.cur_idx

                # ---------------------------
                # PAINT TICK (always publish)
                # ---------------------------
                self.logic_accum += dt
                logic_step = 1.0 / self.logic_hz
                painted_any = False

                while self.logic_accum >= logic_step:
                    self.logic_accum -= logic_step

                    with self.shared.lock:
                        render_idx = self.shared.render_idx
                    self._copy_on_write_if_needed(render_idx)

                    grid = self.shared.buffers[self.cur_idx]
                    self.paint_batch.reset()

                    # one draw call per cursor per tick
                    for c in self.autopilot.cursors:
                        cx, cy, r, k = c.tick(now)
                        if k <= 0:
                            continue
                        painted_any = True
                        if k > self.gw * self.gh:
                            k = self.gw * self.gh
                        val = c.pick_value(self.engine.variant)

                        remaining = k
                        while remaining > 0:
                            space = self.paint_batch.capacity - self.paint_batch.n
                            if space <= 0:
                                self.paint_batch.flush_into(grid)
                                continue
                            take = remaining if remaining < space else space
                            self.paint_batch.add_disk_points(self.disk_pool, cx, cy, r, take, val, self.gw, self.gh)
                            remaining -= take

                    # user commands (batched)
                    for (cx, cy, r, k, val) in self._drain_paint_commands():
                        if k <= 0:
                            continue
                        painted_any = True
                        if k > self.gw * self.gh:
                            k = self.gw * self.gh
                        remaining = k
                        while remaining > 0:
                            space = self.paint_batch.capacity - self.paint_batch.n
                            if space <= 0:
                                self.paint_batch.flush_into(grid)
                                continue
                            take = remaining if remaining < space else space
                            self.paint_batch.add_disk_points(self.disk_pool, cx, cy, r, take, val, self.gw, self.gh)
                            remaining -= take

                    self.paint_batch.flush_into(grid)

                    if painted_any:
                        with self.shared.lock:
                            self.shared.pub_idx = self.cur_idx

                # ---------------------------
                # SIMULATION STEPS (publish)
                # ---------------------------
                self.sim_accum += dt
                step_interval = 1.0 / max(1e-6, float(speed_ips))
                steps = 0
                max_steps = 32

                while self.sim_accum >= step_interval and steps < max_steps:
                    self.sim_accum -= step_interval
                    steps += 1

                    with self.shared.lock:
                        render_idx = self.shared.render_idx
                    out_idx = self._pick_free_idx(self.cur_idx, render_idx)

                    grid_in = self.shared.buffers[self.cur_idx]
                    grid_out = self.shared.buffers[out_idx]
                    self.engine.update_from_to(grid_in, grid_out, colored_counts=self.colored_counts)
                    self.cur_idx = out_idx

                sim_ips = None
                if steps > 0:
                    with self.shared.lock:
                        self.shared.pub_idx = self.cur_idx

                    self.sim_steps += steps
                    t2 = time.perf_counter()
                    if t2 - self.sim_last >= 1.0:
                        dtm = t2 - self.sim_last
                        sim_ips = self.sim_steps / dtm if dtm > 0 else 0.0
                        self.sim_steps = 0
                        self.sim_last = t2

                # metrics
                with self.shared.lock:
                    self.shared.metrics["speed_ips"] = float(speed_ips)
                    self.shared.metrics["colors"] = colors
                    self.shared.metrics["variant"] = self.engine.variant
                    self.shared.metrics["step_count"] = int(self.engine.step_count)
                    if sim_ips is not None:
                        self.shared.metrics["sim_ips"] = float(sim_ips)

                time.sleep(0.001)

        except Exception as e:
            msg = f"[SIM THREAD CRASH] {repr(e)}"
            log_to_temp(msg)
            with self.shared.lock:
                self.shared.metrics["error"] = msg
            # keep thread dead; main will keep rendering last published frame

# =============================================================================
# Debug HUD (optional)
# =============================================================================
class SystemStats:
    def __init__(self):
        self.have_psutil = HAS_PSUTIL
        self.have_nvml = False
        self.nvml_handle = None

        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.have_nvml = True
            except Exception:
                self.have_nvml = False

        self.last_poll = 0.0
        self.cpu = 0.0
        self.mem = 0.0
        self.gpu = None

        if self.have_psutil:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def poll(self, now):
        if now - self.last_poll < 1.0:
            return
        self.last_poll = now

        if self.have_psutil:
            try:
                self.cpu = float(psutil.cpu_percent(interval=None))
                self.mem = float(psutil.virtual_memory().percent)
            except Exception:
                self.cpu = 0.0
                self.mem = 0.0

        if self.have_nvml and self.nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                self.gpu = float(util.gpu)
            except Exception:
                self.gpu = None

class DebugHUD:
    def __init__(self, w, font_name="Consolas", font_size=18):
        self.w = w
        self.font = pygame.font.SysFont(font_name, font_size)
        self.color = (255, 220, 0)
        self.stats = SystemStats()
        self.frames = 0
        self.last_fps_tick = time.time()
        self.frames_per_sec = 0.0

    def note_frame(self):
        self.frames += 1
        now = time.time()
        if now - self.last_fps_tick >= 1.0:
            dt = now - self.last_fps_tick
            self.frames_per_sec = self.frames / dt if dt > 0 else 0.0
            self.frames = 0
            self.last_fps_tick = now

    def draw(self, screen, clock, shared: SharedState):
        now = time.time()
        self.stats.poll(now)

        with shared.lock:
            m = dict(shared.metrics)

        lines = []
        lines.append("CONWAY DEBUG")
        lines.append(f"FPS(clock) : {clock.get_fps():6.1f}")
        lines.append(f"Frames/s   : {self.frames_per_sec:6.1f}")
        lines.append(f"Sim it/s   : {m.get('sim_ips', 0.0):6.1f}")
        lines.append(f"Speed ips  : {m.get('speed_ips', 0.0):6.1f}")
        lines.append(f"Variant    : {m.get('variant','')}")
        lines.append(f"Step       : {m.get('step_count',0)}")

        if self.stats.have_psutil:
            lines.append(f"CPU %      : {self.stats.cpu:6.1f}")
            lines.append(f"Mem %      : {self.stats.mem:6.1f}")
        if self.stats.gpu is not None:
            lines.append(f"GPU %      : {self.stats.gpu:6.1f}")

        err = m.get("error", "")
        if err:
            lines.append("ERROR:")
            lines.append(err[:120])

        rendered = [self.font.render(s, True, self.color) for s in lines]
        max_w = max(s.get_width() for s in rendered) if rendered else 0
        x0 = self.w - max_w - 12
        y = 8
        for surf in rendered:
            screen.blit(surf, (x0, y))
            y += surf.get_height() + 2

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(prefix_chars='-/')
    parser.add_argument('--screensaver', action='store_true', help="Run in screensaver mode")
    parser.add_argument('/s', action='store_true', dest='windows_screensaver', help="Windows Screensaver Launch Flag")
    parser.add_argument('/p', dest='preview_mode', nargs='?', help="Preview Mode")
    parser.add_argument('/c', dest='config_mode', nargs='?', help="Config Mode")
    parser.add_argument('--debug', action='store_true', help="Enable debug HUD in python run")
    args, _unknown = parser.parse_known_args()

    if args.config_mode or args.preview_mode:
        return

    is_screensaver = args.screensaver or args.windows_screensaver
    debug_enabled = (BUILD_DEBUG or args.debug)

    # virtual desktop bbox
    if os.name == "nt":
        rects = get_monitor_rects_windows()
        if rects:
            vx, vy, vw, vh = bounding_box_from_rects(rects)
        else:
            vx, vy, vw, vh = get_virtual_screen_bounds()
            rects = []
    else:
        vx, vy, vw, vh = get_virtual_screen_bounds()
        rects = []

    global GRID_SIZE
    GRID_SIZE = choose_grid_size_explosive(vw, vh)

    pygame.init()

    def set_mode_safe(size, flags, want_vsync=True):
        try:
            return pygame.display.set_mode(size, flags, vsync=1 if want_vsync else 0)
        except TypeError:
            return pygame.display.set_mode(size, flags)

    if is_screensaver:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{vx},{vy}"
        screen = set_mode_safe((vw, vh), pygame.NOFRAME | pygame.DOUBLEBUF, want_vsync=True)
        pygame.display.set_caption("Conway Multi-Monitor Screensaver")
        pygame.mouse.set_visible(True)
        w, h = vw, vh
        monitor_rects = rects
        virtual_origin = (vx, vy)
    else:
        w, h = 1280, 720
        screen = set_mode_safe((w, h), pygame.RESIZABLE | pygame.DOUBLEBUF, want_vsync=True)
        pygame.display.set_caption("Conway Debug")
        monitor_rects = []
        virtual_origin = (0, 0)

    clock = pygame.time.Clock()
    gw, gh = max(1, w // GRID_SIZE), max(1, h // GRID_SIZE)

    buffers = [
        np.zeros((gh, gw), dtype=np.int8),
        np.zeros((gh, gw), dtype=np.int8),
        np.zeros((gh, gw), dtype=np.int8),
    ]
    shared = SharedState(buffers)

    sim = SimWorker(shared, gw, gh, monitor_rects, virtual_origin, debug_enabled=debug_enabled)
    sim.start()

    # user draw controls (same semantics as autopilot)
    user_brush_size = 24
    user_fill = 0.010  # per draw call fill fraction; tune taste
    user_palette_sel = 1

    # render buffers
    render_surface = pygame.Surface((gw, gh), flags=0, depth=32)
    packed_buf = np.empty((gh, gw), dtype=np.uint32)
    packed_palette = np.zeros(5, dtype=np.uint32)
    need_scale = (gw != w) or (gh != h)
    scaled_surface = pygame.Surface((w, h), flags=0, depth=32) if need_scale else None

    hud = DebugHUD(w) if debug_enabled else None

    running = True
    while running:
        keep_display_awake()
        dt = clock.tick(FPS_MAX) / 1000.0
        if dt < 0.0:
            dt = 0.0
        if dt > 0.25:
            dt = 0.25

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            if is_screensaver:
                if event.type == pygame.MOUSEWHEEL:
                    max_user = max(10, int(min(gw, gh) / 3))
                    user_brush_size = int(clamp(user_brush_size + event.y, 1, max_user))
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: user_palette_sel = 1
                    if event.key == pygame.K_2: user_palette_sel = 2
                    if event.key == pygame.K_3: user_palette_sel = 3
                    if event.key == pygame.K_4: user_palette_sel = 4

        # one user draw call per frame
        if is_screensaver and (pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]):
            mx, my = pygame.mouse.get_pos()
            gx, gy = mx // GRID_SIZE, my // GRID_SIZE
            r = max(1, int(user_brush_size) - 1)
            k = points_for_fill_fraction(r, user_fill)
            if k <= 0:
                k = 1
            if k > gw * gh:
                k = gw * gh
            val = int(user_palette_sel) if pygame.mouse.get_pressed()[0] else 0

            with shared.paint_lock:
                shared.paint_q.append((int(gx), int(gy), int(r), int(k), int(val)))

        # grab published buffer
        with shared.lock:
            pub_idx = shared.pub_idx
            shared.render_idx = pub_idx
            colors = shared.metrics.get("colors", DEFAULT_COLORS)

        grid = shared.buffers[pub_idx]

        # render
        packed_palette[0] = render_surface.map_rgb(colors[0])
        packed_palette[1] = render_surface.map_rgb(colors[1])
        packed_palette[2] = render_surface.map_rgb(colors[2])
        packed_palette[3] = render_surface.map_rgb(colors[3])
        packed_palette[4] = render_surface.map_rgb(colors[4])

        np.take(packed_palette, grid, out=packed_buf)
        pygame.surfarray.blit_array(render_surface, packed_buf.T)

        screen.fill(COLOR_BG)
        if need_scale:
            pygame.transform.scale(render_surface, (w, h), scaled_surface)
            screen.blit(scaled_surface, (0, 0))
        else:
            screen.blit(render_surface, (0, 0))

        if hud is not None:
            hud.note_frame()
            hud.draw(screen, clock, shared)

        pygame.display.flip()

    with shared.lock:
        shared.stop = True
    sim.join(timeout=1.0)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
