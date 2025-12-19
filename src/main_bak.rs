// src/main.rs
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use pollster::block_on;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

#[cfg(windows)]
use windows_sys::Win32::{
    Foundation::{LPARAM, RECT},
    Graphics::Gdi::{EnumDisplayMonitors, GetMonitorInfoW, HDC, HMONITOR, MONITORINFO},
    UI::WindowsAndMessaging::{
        GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
    },
};

// -----------------------------
// Embedded WGSL shader (fullscreen blit)
// -----------------------------
const BLIT_WGSL: &str = r#"
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 2.0),
    vec2<f32>(2.0, 0.0),
    vec2<f32>(0.0, 0.0)
  );

  var o: VSOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.uv  = uv[vi];
  return o;
}

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;

@fragment
fn fs_main(i: VSOut) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, i.uv);
}
"#;

// -----------------------------
// Helpers
// -----------------------------
fn clamp<T: PartialOrd>(x: T, lo: T, hi: T) -> T {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

fn smoothstep(t: f32) -> f32 {
    let t = clamp(t, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn choose_grid_size_explosive(vw: i32, vh: i32) -> u32 {
    let short = (vw.abs().min(vh.abs()).max(1)) as f32;
    let gs = (short / 1000.0).round() as i32;
    gs.max(1) as u32
}

// Python semantics:
// p ≈ 1 - exp(-k/A) => k = -A ln(1-p), A = pi r^2
fn points_for_fill_fraction(radius_cells: i32, fill_fraction: f32) -> usize {
    let r = radius_cells.max(1) as f32;
    let p = clamp(fill_fraction, 0.0, 0.999_999);
    if p <= 0.0 {
        return 0;
    }
    let area = std::f32::consts::PI * (r * r);
    let k = (-area * (1.0 - p).ln()) as i32;
    k.max(0) as usize
}

#[derive(Default, Clone, Copy)]
struct InputState {
    cursor_px: Option<(f32, f32)>,
    left_down: bool,
    right_down: bool,
    wheel_delta: f32,
}

// -----------------------------
// Windows multi-monitor bounds
// -----------------------------
#[cfg(windows)]
#[derive(Clone, Copy, Debug)]
struct MonRect {
    l: i32,
    t: i32,
    r: i32,
    b: i32,
}

#[cfg(windows)]
fn monitor_rects() -> Vec<MonRect> {
    unsafe extern "system" fn cb(
        hmon: HMONITOR,
        _hdc: HDC,
        _rc: *mut RECT,
        lparam: LPARAM,
    ) -> i32 {
        unsafe {
            let out = &mut *(lparam as *mut Vec<MonRect>);
            let mut mi = MONITORINFO {
                cbSize: std::mem::size_of::<MONITORINFO>() as u32,
                rcMonitor: RECT {
                    left: 0,
                    top: 0,
                    right: 0,
                    bottom: 0,
                },
                rcWork: RECT {
                    left: 0,
                    top: 0,
                    right: 0,
                    bottom: 0,
                },
                dwFlags: 0,
            };
            if GetMonitorInfoW(hmon, &mut mi as *mut MONITORINFO) != 0 {
                let r = mi.rcMonitor;
                out.push(MonRect {
                    l: r.left,
                    t: r.top,
                    r: r.right,
                    b: r.bottom,
                });
            }
            1
        }
    }

    let mut rects: Vec<MonRect> = Vec::new();
    unsafe {
        EnumDisplayMonitors(
            std::ptr::null_mut(),
            std::ptr::null(),
            Some(cb),
            (&mut rects as *mut Vec<MonRect>) as isize,
        );
    }
    rects
}

#[cfg(windows)]
fn bbox(rects: &[MonRect]) -> (i32, i32, i32, i32) {
    let mut l = i32::MAX;
    let mut t = i32::MAX;
    let mut r = i32::MIN;
    let mut b = i32::MIN;
    for m in rects {
        l = l.min(m.l);
        t = t.min(m.t);
        r = r.max(m.r);
        b = b.max(m.b);
    }
    (l, t, r - l, b - t)
}

#[cfg(windows)]
fn virtual_bounds_windows() -> (i32, i32, i32, i32, Vec<MonRect>) {
    let rects = monitor_rects();
    if !rects.is_empty() {
        let (vx, vy, vw, vh) = bbox(&rects);
        return (vx, vy, vw.max(1), vh.max(1), rects);
    }
    unsafe {
        let vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
        let vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
        let vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        let vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        if vw > 0 && vh > 0 {
            return (vx, vy, vw, vh, vec![]);
        }
    }
    (0, 0, 1280, 720, vec![])
}

// -----------------------------
// Smooth transitions (match Python: next_change randomized)
// -----------------------------
#[derive(Clone, Copy)]
struct Transition {
    cur: f32,
    start: f32,
    target: f32,
    t0: f32,
    dur: f32,
    next_change: f32,
    lo: f32,
    hi: f32,
    min_dur: f32,
    max_dur: f32,
}
impl Transition {
    fn new<R: Rng + ?Sized>(rng: &mut R, initial: f32, lo: f32, hi: f32, min_dur: f32, max_dur: f32) -> Self {
        let now = 0.0;
        let next = now + rng.random_range(min_dur..max_dur);
        Self {
            cur: initial,
            start: initial,
            target: initial,
            t0: now,
            dur: 1.0,
            next_change: next,
            lo,
            hi,
            min_dur,
            max_dur,
        }
    }

    fn set_target(&mut self, now: f32, v: f32, dur: f32) {
        let v = clamp(v, self.lo, self.hi);
        self.start = self.cur;
        self.target = v;
        self.t0 = now;
        self.dur = dur.max(0.001);
        self.next_change = now + self.dur;
    }

    fn maybe_new_target<R: Rng + ?Sized, F: FnOnce(&mut R) -> f32>(&mut self, now: f32, rng: &mut R, pick: F) {
        if now >= self.next_change {
            let dur = rng.random_range(self.min_dur..self.max_dur);
            let v = pick(rng);
            self.set_target(now, v, dur);
        }
    }

    fn update(&mut self, now: f32) -> f32 {
        let t = (now - self.t0) / self.dur;
        let u = smoothstep(t);
        self.cur = self.start + (self.target - self.start) * u;
        self.cur = clamp(self.cur, self.lo, self.hi);
        self.cur
    }
}

#[derive(Clone, Copy)]
struct ColorTransition {
    cur: [u8; 3],
    start: [u8; 3],
    target: [u8; 3],
    t0: f32,
    dur: f32,
    next_change: f32,
    min_dur: f32,
    max_dur: f32,
}
impl ColorTransition {
    fn new<R: Rng + ?Sized>(rng: &mut R, rgb: [u8; 3], min_dur: f32, max_dur: f32) -> Self {
        let now = 0.0;
        let next = now + rng.random_range(min_dur..max_dur);
        Self {
            cur: rgb,
            start: rgb,
            target: rgb,
            t0: now,
            dur: 1.0,
            next_change: next,
            min_dur,
            max_dur,
        }
    }

    fn set_target(&mut self, now: f32, rgb: [u8; 3], dur: f32) {
        self.start = self.cur;
        self.target = rgb;
        self.t0 = now;
        self.dur = dur.max(0.001);
        self.next_change = now + self.dur;
    }

    fn maybe_new_target<R: Rng + ?Sized, F: FnOnce(&mut R) -> [u8; 3]>(&mut self, now: f32, rng: &mut R, pick: F) {
        if now >= self.next_change {
            let dur = rng.random_range(self.min_dur..self.max_dur);
            let rgb = pick(rng);
            self.set_target(now, rgb, dur);
        }
    }

    fn update(&mut self, now: f32) -> [u8; 3] {
        let t = (now - self.t0) / self.dur;
        let u = smoothstep(t);
        let mut out = [0u8; 3];
        for i in 0..3 {
            let a = self.start[i] as f32;
            let b = self.target[i] as f32;
            out[i] = (a + (b - a) * u).round().clamp(0.0, 255.0) as u8;
        }
        self.cur = out;
        out
    }
}

// -----------------------------
// Variants (Python + “screensaver-grade” additions)
// -----------------------------
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Variant {
    Standard,
    HighLife,
    DayNight,
    Seeds,
    Immigration,
    QuadLife,
    Generations,
    BriansBrain,
    LargerThanLife,
    Margolus,

    // additions (life-like)
    Maze,
    Coral,
    Anneal,
    Diamoeba,
    Replicator,
    TwoByTwo,

    // additions (multi-state / continuous-ish)
    Cyclic,
    GreenbergHastings,
    GrayScott,
}

impl Variant {
    fn all() -> &'static [Variant] {
        &[
            Variant::Standard,
            Variant::HighLife,
            Variant::DayNight,
            Variant::Seeds,
            Variant::Immigration,
            Variant::QuadLife,
            Variant::Generations,
            Variant::BriansBrain,
            Variant::LargerThanLife,
            Variant::Margolus,
            Variant::Maze,
            Variant::Coral,
            Variant::Anneal,
            Variant::Diamoeba,
            Variant::Replicator,
            Variant::TwoByTwo,
            Variant::Cyclic,
            Variant::GreenbergHastings,
            Variant::GrayScott,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Variant::Standard => "STANDARD",
            Variant::HighLife => "HIGHLIFE",
            Variant::DayNight => "DAY-NIGHT",
            Variant::Seeds => "SEEDS",
            Variant::Immigration => "IMMIGRATION",
            Variant::QuadLife => "QUADLIFE",
            Variant::Generations => "GENERATIONS",
            Variant::BriansBrain => "BRIAN'S BRAIN",
            Variant::LargerThanLife => "LARGER THAN LIFE",
            Variant::Margolus => "MARGOLUS",
            Variant::Maze => "MAZE",
            Variant::Coral => "CORAL",
            Variant::Anneal => "ANNEAL",
            Variant::Diamoeba => "DIAMOEBA",
            Variant::Replicator => "REPLICATOR",
            Variant::TwoByTwo => "TWOBYTWO",
            Variant::Cyclic => "CYCLIC",
            Variant::GreenbergHastings => "GREENBERGHASTING",
            Variant::GrayScott => "GRAY-SCOTT",
        }
    }
}


// -----------------------------
// Simulation core
// -----------------------------
#[derive(Copy, Clone)]
struct LifeRule {
    survive_mask: u16,
    birth_mask: u16,
}
impl LifeRule {
    fn from_sets(birth: &[u8], survive: &[u8]) -> Self {
        let mut b = 0u16;
        let mut s = 0u16;
        for &n in birth {
            b |= 1 << (n as u16);
        }
        for &n in survive {
            s |= 1 << (n as u16);
        }
        Self {
            survive_mask: s,
            birth_mask: b,
        }
    }
    fn survive(&self, n: i32) -> bool {
        if !(0..=15).contains(&n) {
            return false;
        }
        (self.survive_mask & (1 << n)) != 0
    }
    fn birth(&self, n: i32) -> bool {
        if !(0..=15).contains(&n) {
            return false;
        }
        (self.birth_mask & (1 << n)) != 0
    }
}

#[derive(Copy, Clone)]
enum MaskMode {
    Alive,   // v != 0
    Eq(u8),  // v == k
}
impl MaskMode {
    #[inline]
    fn to01(self, v: u8) -> u8 {
        match self {
            MaskMode::Alive => (v != 0) as u8,
            MaskMode::Eq(k) => (v == k) as u8,
        }
    }
}

#[derive(Copy, Clone)]
struct SyncU16Ptr(*mut u16);
unsafe impl Send for SyncU16Ptr {}
unsafe impl Sync for SyncU16Ptr {}
impl SyncU16Ptr {
    #[inline]
    unsafe fn add(self, off: usize) -> *mut u16 {
        unsafe { self.0.add(off) }
    }
}


struct Sim {
    w: usize,
    h: usize,
    n: usize,

    variant: Variant,
    parity: u8,

    cur: Vec<u8>,
    next: Vec<u8>,

    // rolling sums
    hsum: Vec<u16>,
    vsum: Vec<u16>,

    // scratch (avoid allocations for colored neighbor counts)
    tmp1: Vec<u16>,
    tmp2: Vec<u16>,
    tmp3: Vec<u16>,
    tmp4: Vec<u16>,
    tmp5: Vec<u16>,

    // Generations (Python MAX_DECAY=4)
    gen_decay_max: u8,

    // LTL (Python defaults)
    ltl_range: i32,
    ltl_birth: (i32, i32),
    ltl_survive: (i32, i32),

    // Cyclic
    cyclic_states: u8,

    // Greenberg–Hastings
    gh_refrac_max: u8,

    // Gray–Scott
    gs_u: Vec<f32>,
    gs_v: Vec<f32>,
    gs_u2: Vec<f32>,
    gs_v2: Vec<f32>,
}

impl Sim {
    fn new(w: usize, h: usize) -> Self {
        let n = w * h;
        Self {
            w,
            h,
            n,
            variant: Variant::Standard,
            parity: 0,
            cur: vec![0; n],
            next: vec![0; n],
            hsum: vec![0; n],
            vsum: vec![0; n],
            tmp1: vec![0; n],
            tmp2: vec![0; n],
            tmp3: vec![0; n],
            tmp4: vec![0; n],
            tmp5: vec![0; n],

            gen_decay_max: 4,
            ltl_range: 5,
            ltl_birth: (34, 45),
            ltl_survive: (33, 57),

            cyclic_states: 12,
            gh_refrac_max: 10,

            gs_u: vec![1.0; n],
            gs_v: vec![0.0; n],
            gs_u2: vec![1.0; n],
            gs_v2: vec![0.0; n],
        }
    }

    fn clear(&mut self) {
        self.cur.fill(0);
        self.next.fill(0);
        self.hsum.fill(0);
        self.vsum.fill(0);
        self.tmp1.fill(0);
        self.tmp2.fill(0);
        self.tmp3.fill(0);
        self.tmp4.fill(0);
        self.tmp5.fill(0);
        self.gs_u.fill(1.0);
        self.gs_v.fill(0.0);
        self.gs_u2.fill(1.0);
        self.gs_v2.fill(0.0);
        self.parity = 0;
    }

    // Python-like seeding:
    // - briansbrain: 0/2 with p=0.92/0.08
    // - immigration: 0/1/2 with p=0.82/0.09/0.09
    // - otherwise:   0/1 with p=0.86/0.14
    fn seed_random<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        match self.variant {
            Variant::BriansBrain => {
                for v in &mut self.cur {
                    let r: f32 = rng.random();
                    *v = if r < 0.92 { 0 } else { 2 };
                }
            }
            Variant::Immigration => {
                for v in &mut self.cur {
                    let r: f32 = rng.random();
                    *v = if r < 0.82 {
                        0
                    } else if r < 0.91 {
                        1
                    } else {
                        2
                    };
                }
            }
            Variant::Cyclic => {
                let k = self.cyclic_states.max(2);
                for v in &mut self.cur {
                    *v = rng.random_range(0..k) as u8;
                }
            }
            Variant::GreenbergHastings => {
                let m = self.gh_refrac_max.max(3);
                for v in &mut self.cur {
                    let r: f32 = rng.random();
                    *v = if r < 0.95 {
                        0
                    } else if r < 0.98 {
                        1
                    } else {
                        rng.random_range(2..=m) as u8
                    };
                }
            }
            Variant::GrayScott => {
                self.gs_u.fill(1.0);
                self.gs_v.fill(0.0);
                for _ in 0..10 {
                    let cx = rng.random_range(0..self.w) as i32;
                    let cy = rng.random_range(0..self.h) as i32;
                    let rad = rng.random_range(20..80);
                    self.seed_blob_gray_scott(cx, cy, rad);
                }
                self.update_gray_scott_cur_from_v();
            }
            _ => {
                for v in &mut self.cur {
                    let r: f32 = rng.random();
                    *v = if r < 0.86 { 0 } else { 1 };
                }
            }
        }
    }

    fn seed_blob_gray_scott(&mut self, cx: i32, cy: i32, radius: i32) {
        let w = self.w as i32;
        let h = self.h as i32;
        let r2 = radius * radius;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let x = (cx + dx).rem_euclid(w) as usize;
                let y = (cy + dy).rem_euclid(h) as usize;
                let i = y * self.w + x;
                self.gs_u[i] = 0.5;
                self.gs_v[i] = 0.25;
            }
        }
    }

    fn update_gray_scott_cur_from_v(&mut self) {
        for i in 0..self.n {
            let v = self.gs_v[i];
            self.cur[i] = if v < 0.02 {
                0
            } else if v < 0.08 {
                1
            } else if v < 0.16 {
                2
            } else if v < 0.28 {
                3
            } else {
                4
            };
        }
    }

    // Rolling Moore sum for mask, radius r (wrap).
    fn moore_sum_mask(&mut self, r: i32, mode: MaskMode) {
        let w = self.w;
        let h = self.h;
        let r = r as i32;

        let cur = &self.cur;
        self.hsum
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, hrow)| {
                let y = y as i32;
                let base = (y as usize) * w;

                let mut s: u16 = 0;
                for dx in -r..=r {
                    let x = (dx).rem_euclid(w as i32) as usize;
                    s += mode.to01(cur[base + x]) as u16;
                }
                hrow[0] = s;

                for x in 1..w {
                    let x_add = ((x as i32 + r).rem_euclid(w as i32)) as usize;
                    let x_sub = ((x as i32 - r - 1).rem_euclid(w as i32)) as usize;
                    s += mode.to01(cur[base + x_add]) as u16;
                    s -= mode.to01(cur[base + x_sub]) as u16;
                    hrow[x] = s;
                }
            });

        let vsum_ptr = SyncU16Ptr(self.vsum.as_mut_ptr());
        let hsum = &self.hsum;

        (0..w).into_par_iter().for_each(|x| {
            unsafe {
                let mut s: u16 = 0;
                for dy in -r..=r {
                    let y = (dy).rem_euclid(h as i32) as usize;
                    s += hsum[y * w + x];
                }
                *vsum_ptr.add(0 * w + x) = s;

                for y in 1..h {
                    let y_add = ((y as i32 + r).rem_euclid(h as i32)) as usize;
                    let y_sub = ((y as i32 - r - 1).rem_euclid(h as i32)) as usize;
                    s += hsum[y_add * w + x];
                    s -= hsum[y_sub * w + x];
                    *vsum_ptr.add(y * w + x) = s;
                }
            }
        });
    }

    fn step_life_like(&mut self, rule: LifeRule) {
        self.moore_sum_mask(1, MaskMode::Alive);
        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let alive = cur[i] != 0;
                let sum_including_self = self.vsum[i] as i32;
                let n = sum_including_self - (alive as i32);
                let born = !alive && rule.birth(n);
                let surv = alive && rule.survive(n);
                nrow[x] = if born || surv { 1 } else { 0 };
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_immigration(&mut self) {
        self.moore_sum_mask(1, MaskMode::Alive);
        self.tmp1.copy_from_slice(&self.vsum); // total

        self.moore_sum_mask(1, MaskMode::Eq(1));
        self.tmp2.copy_from_slice(&self.vsum); // c1

        self.moore_sum_mask(1, MaskMode::Eq(2));
        self.tmp3.copy_from_slice(&self.vsum); // c2

        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;

        let total = &self.tmp1;
        let c1 = &self.tmp2;
        let c2 = &self.tmp3;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let v = cur[i];
                let alive = v != 0;
                let tn = (total[i] as i32) - (alive as i32);
                let survive = alive && (tn == 2 || tn == 3);
                let born = !alive && tn == 3;

                if survive {
                    nrow[x] = v;
                } else if born {
                    let n1 = c1[i] as i32;
                    let n2 = c2[i] as i32;
                    nrow[x] = if n2 > n1 { 2 } else { 1 };
                } else {
                    nrow[x] = 0;
                }
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_quadlife(&mut self) {
        self.moore_sum_mask(1, MaskMode::Alive);
        self.tmp1.copy_from_slice(&self.vsum); // total

        self.moore_sum_mask(1, MaskMode::Eq(1));
        self.tmp2.copy_from_slice(&self.vsum); // c1
        self.moore_sum_mask(1, MaskMode::Eq(2));
        self.tmp3.copy_from_slice(&self.vsum); // c2
        self.moore_sum_mask(1, MaskMode::Eq(3));
        self.tmp4.copy_from_slice(&self.vsum); // c3
        self.moore_sum_mask(1, MaskMode::Eq(4));
        self.tmp5.copy_from_slice(&self.vsum); // c4

        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;

        let total = &self.tmp1;
        let c1 = &self.tmp2;
        let c2 = &self.tmp3;
        let c3 = &self.tmp4;
        let c4 = &self.tmp5;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let v = cur[i];
                let alive = v != 0;
                let tn = (total[i] as i32) - (alive as i32);
                let survive = alive && (tn == 2 || tn == 3);
                let born = !alive && tn == 3;

                if survive {
                    nrow[x] = v;
                } else if born {
                    let a = c1[i];
                    let b = c2[i];
                    let c = c3[i];
                    let d = c4[i];
                    let mx = a.max(b).max(c).max(d);
                    let mut out = 1;
                    if b == mx { out = 2; }
                    if c == mx { out = 3; }
                    if d == mx { out = 4; }
                    nrow[x] = out;
                } else {
                    nrow[x] = 0;
                }
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_generations(&mut self) {
        self.moore_sum_mask(1, MaskMode::Alive);
        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;
        let maxd = self.gen_decay_max;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let v = cur[i];
                let alive = v == 1;
                let any = v != 0;
                let sum = self.vsum[i] as i32;
                let n = sum - (any as i32);

                if alive {
                    let stays = n == 2 || n == 3;
                    nrow[x] = if stays { 1 } else { 2 };
                } else if v >= 2 {
                    let nv = v.saturating_add(1);
                    nrow[x] = if nv > maxd { 0 } else { nv };
                } else {
                    nrow[x] = if n == 3 { 1 } else { 0 };
                }
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    // Match Python encoding:
    // 0 rest, 2 firing, 1 refractory (one step).
    fn step_brians_brain(&mut self) {
        self.moore_sum_mask(1, MaskMode::Eq(2)); // count firing neighbors
        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let v = cur[i];
                if v == 0 {
                    let n = self.vsum[i] as i32;
                    nrow[x] = if n == 2 { 2 } else { 0 };
                } else if v == 2 {
                    nrow[x] = 1;
                } else {
                    // v == 1 -> back to rest
                    nrow[x] = 0;
                }
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_larger_than_life(&mut self) {
        let r = self.ltl_range;
        self.moore_sum_mask(r, MaskMode::Alive);

        let (b0, b1) = self.ltl_birth;
        let (s0, s1) = self.ltl_survive;

        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let alive = cur[i] != 0;
                let sum = self.vsum[i] as i32;
                let n = sum - (alive as i32);

                let born = !alive && (n >= b0 && n <= b1);
                let surv = alive && (n >= s0 && n <= s1);
                nrow[x] = if born || surv { 1 } else { 0 };
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_margolus(&mut self) {
        let w = self.w;
        let h = self.h;
        let shift = self.parity as usize;

        self.next.copy_from_slice(&self.cur);

        for y0 in (0..h).step_by(2) {
            for x0 in (0..w).step_by(2) {
                let y = (y0 + shift) % h;
                let x = (x0 + shift) % w;

                let x1 = (x + 1) % w;
                let y1 = (y + 1) % h;

                let i00 = y * w + x;
                let i01 = y * w + x1;
                let i10 = y1 * w + x;
                let i11 = y1 * w + x1;

                let a = self.cur[i00];
                let b = self.cur[i01];
                let c = self.cur[i11];
                let d = self.cur[i10];

                let sum = (a != 0) as u8 + (b != 0) as u8 + (c != 0) as u8 + (d != 0) as u8;

                if sum == 1 || sum == 3 {
                    // matches Python block rotation
                    self.next[i00] = d;
                    self.next[i01] = a;
                    self.next[i11] = b;
                    self.next[i10] = c;
                } else {
                    self.next[i00] = a;
                    self.next[i01] = b;
                    self.next[i11] = c;
                    self.next[i10] = d;
                }
            }
        }

        self.parity ^= 1;
        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_cyclic(&mut self) {
        let k = self.cyclic_states.max(2) as u16;
        let w = self.w;
        let h = self.h;
        let cur = &self.cur;
        let next = &mut self.next;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as i32;
            let ym1 = (y - 1).rem_euclid(h as i32);
            let yp1 = (y + 1).rem_euclid(h as i32);
            for x in 0..w {
                let x = x as i32;
                let xm1 = (x - 1).rem_euclid(w as i32);
                let xp1 = (x + 1).rem_euclid(w as i32);

                let i = (y as usize) * w + (x as usize);
                let v = cur[i];
                let want = (((v as u16) + 1) % k) as u8;

                let coords = [
                    (xm1, ym1),
                    (x, ym1),
                    (xp1, ym1),
                    (xm1, y),
                    (xp1, y),
                    (xm1, yp1),
                    (x, yp1),
                    (xp1, yp1),
                ];

                let mut hit = false;
                for (xx, yy) in coords {
                    let j = (yy as usize) * w + (xx as usize);
                    if cur[j] == want {
                        hit = true;
                        break;
                    }
                }

                nrow[x as usize] = if hit { want } else { v };
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_greenberg_hastings(&mut self) {
        // 0 rest, 1 excited, 2..m refractory
        self.moore_sum_mask(1, MaskMode::Eq(1)); // count excited neighbors
        let m = self.gh_refrac_max.max(3);

        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let v = cur[i];
                if v == 0 {
                    let n = self.vsum[i];
                    nrow[x] = if n > 0 { 1 } else { 0 };
                } else if v == 1 {
                    nrow[x] = 2;
                } else if v < m {
                    nrow[x] = v + 1;
                } else {
                    nrow[x] = 0;
                }
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_gray_scott(&mut self, dt: f32) {
        // conservative, stable defaults (you can randomize later if desired)
        let du = 0.16;
        let dv = 0.08;
        let f = 0.035;
        let k = 0.065;

        let w = self.w;
        let h = self.h;
        let u = &self.gs_u;
        let v = &self.gs_v;
        let u2 = &mut self.gs_u2;
        let v2 = &mut self.gs_v2;

        u2.par_chunks_mut(w)
            .zip(v2.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (u2row, v2row))| {
                let ym1 = if y == 0 { h - 1 } else { y - 1 };
                let yp1 = if y + 1 == h { 0 } else { y + 1 };
                for x in 0..w {
                    let xm1 = if x == 0 { w - 1 } else { x - 1 };
                    let xp1 = if x + 1 == w { 0 } else { x + 1 };

                    let i = y * w + x;

                    let u_c = u[i];
                    let v_c = v[i];

                    let u_n = u[ym1 * w + x];
                    let u_s = u[yp1 * w + x];
                    let u_w = u[y * w + xm1];
                    let u_e = u[y * w + xp1];

                    let v_n = v[ym1 * w + x];
                    let v_s = v[yp1 * w + x];
                    let v_w = v[y * w + xm1];
                    let v_e = v[y * w + xp1];

                    let lap_u = u_n + u_s + u_w + u_e - 4.0 * u_c;
                    let lap_v = v_n + v_s + v_w + v_e - 4.0 * v_c;

                    let uvv = u_c * v_c * v_c;
                    let du_dt = du * lap_u - uvv + f * (1.0 - u_c);
                    let dv_dt = dv * lap_v + uvv - (f + k) * v_c;

                    u2row[x] = (u_c + du_dt * dt).clamp(0.0, 1.0);
                    v2row[x] = (v_c + dv_dt * dt).clamp(0.0, 1.0);
                }
            });

        std::mem::swap(&mut self.gs_u, &mut self.gs_u2);
        std::mem::swap(&mut self.gs_v, &mut self.gs_v2);
        self.update_gray_scott_cur_from_v();
    }

    fn step(&mut self, dt_for_gs: f32) {
        match self.variant {
            Variant::Standard => self.step_life_like(LifeRule::from_sets(&[3], &[2, 3])),
            Variant::HighLife => self.step_life_like(LifeRule::from_sets(&[3, 6], &[2, 3])),
            Variant::DayNight => self.step_life_like(LifeRule::from_sets(&[3, 6, 7, 8], &[3, 4, 6, 7, 8])),
            Variant::Seeds => self.step_life_like(LifeRule::from_sets(&[2], &[])),

            Variant::Maze => self.step_life_like(LifeRule::from_sets(&[3], &[1, 2, 3, 4, 5])),
            Variant::Coral => self.step_life_like(LifeRule::from_sets(&[3], &[4, 5, 6, 7, 8])),
            Variant::Anneal => self.step_life_like(LifeRule::from_sets(&[4, 6, 7, 8], &[3, 5, 6, 7, 8])),
            Variant::Diamoeba => self.step_life_like(LifeRule::from_sets(&[3, 5, 6, 7, 8], &[5, 6, 7, 8])),
            Variant::Replicator => self.step_life_like(LifeRule::from_sets(&[1, 3, 5, 7], &[1, 3, 5, 7])),
            Variant::TwoByTwo => self.step_life_like(LifeRule::from_sets(&[3, 6], &[1, 2, 5])),

            Variant::Immigration => self.step_immigration(),
            Variant::QuadLife => self.step_quadlife(),
            Variant::Generations => self.step_generations(),
            Variant::BriansBrain => self.step_brians_brain(),
            Variant::LargerThanLife => self.step_larger_than_life(),
            Variant::Margolus => self.step_margolus(),

            Variant::Cyclic => self.step_cyclic(),
            Variant::GreenbergHastings => self.step_greenberg_hastings(),
            Variant::GrayScott => self.step_gray_scott(dt_for_gs),
        }
    }
}

// -----------------------------
// Disk sampling (uniform) — fast, no trig in hot path
// -----------------------------
struct RandomDiskPool {
    cos: Vec<f32>,
    sin: Vec<f32>,
    su: Vec<f32>, // sqrt(u)
    idx: usize,
}
impl RandomDiskPool {
    fn new<R: Rng + ?Sized>(rng: &mut R, size: usize) -> Self {
        let mut cos = Vec::with_capacity(size);
        let mut sin = Vec::with_capacity(size);
        let mut su = Vec::with_capacity(size);

        for _ in 0..size {
            let theta = rng.random_range(0.0..std::f32::consts::TAU);
            let u: f32 = rng.random();
            let suv = u.sqrt();
            cos.push(theta.cos());
            sin.push(theta.sin());
            su.push(suv);
        }

        Self { cos, sin, su, idx: 0 }
    }

    #[inline]
    fn next(&mut self) -> (f32, f32, f32) {
        let i = self.idx;
        self.idx += 1;
        if self.idx >= self.cos.len() {
            self.idx = 0;
        }
        (self.cos[i], self.sin[i], self.su[i])
    }
}

// Paint k points uniformly inside a disk (Python semantics).
fn paint_disk_points(sim: &mut Sim, pool: &mut RandomDiskPool, cx: i32, cy: i32, r: i32, k: usize, value: u8) {
    if k == 0 {
        return;
    }
    let w = sim.w as i32;
    let h = sim.h as i32;
    let rr = r.max(1) as f32;

    for _ in 0..k {
        let (c, s, su) = pool.next();
        let fx = cx as f32 + c * su * rr;
        let fy = cy as f32 + s * su * rr;
        let x = (fx.round() as i32).rem_euclid(w) as usize;
        let y = (fy.round() as i32).rem_euclid(h) as usize;
        sim.cur[y * sim.w + x] = value;
    }
}

// -----------------------------
// Autopilot (Python-feel)
// -----------------------------
struct AutoCursor {
    gw: usize,
    gh: usize,
    max_brush: i32,

    x: Transition,
    y: Transition,
    brush: Transition,
    fill: Transition,  // per-tick fill fraction
    layer: Transition, // multiplier for k
}

impl AutoCursor {
    fn new<R: Rng + ?Sized>(rng: &mut R, gw: usize, gh: usize, max_brush: i32, start_xy: Option<(f32, f32)>) -> Self {
        let (sx, sy) = start_xy.unwrap_or((gw as f32 / 2.0, gh as f32 / 2.0));
        let x = Transition::new(rng, sx, 0.0, (gw as f32 - 1.0).max(0.0), 4.0, 10.0);
        let y = Transition::new(rng, sy, 0.0, (gh as f32 - 1.0).max(0.0), 4.0, 10.0);

        let b0 = 20.0_f32.min(max_brush as f32);
        let brush = Transition::new(rng, b0, 10.0, max_brush as f32, 4.0, 10.0);

        let fill = Transition::new(rng, 0.0060, 0.0010, 0.0200, 4.0, 10.0);
        let layer = Transition::new(rng, 1.0, 0.5, 1.5, 4.0, 10.0);

        Self { gw, gh, max_brush, x, y, brush, fill, layer }
    }

    fn pick_value<R: Rng + ?Sized>(&self, rng: &mut R, variant: Variant, sim: &Sim) -> u8 {
        match variant {
            Variant::Immigration => if rng.random::<f32>() < 0.5 { 1 } else { 2 },
            Variant::QuadLife => rng.random_range(1..=4),
            Variant::BriansBrain => 2,
            Variant::Cyclic => {
                let k = sim.cyclic_states.max(2);
                if k <= 2 { 1 } else { rng.random_range(1..k) as u8 } // avoid injecting background
            }
            Variant::GreenbergHastings => 1, // excite
            _ => 1,
        }
    }

    // One paint call per tick, like Python.
    fn tick<R: Rng + ?Sized>(&mut self, now: f32, rng: &mut R) -> (i32, i32, i32, usize) {
        self.x.maybe_new_target(now, rng, |r| r.random_range(0.0..(self.gw as f32 - 1.0).max(1.0)));
        self.y.maybe_new_target(now, rng, |r| r.random_range(0.0..(self.gh as f32 - 1.0).max(1.0)));
        self.brush.maybe_new_target(now, rng, |r| r.random_range(10.0..(self.max_brush as f32)));
        self.fill.maybe_new_target(now, rng, |r| r.random_range(0.0010..0.0200));
        self.layer.maybe_new_target(now, rng, |r| r.random_range(0.5..1.5));

        let x = self.x.update(now).round() as i32;
        let y = self.y.update(now).round() as i32;
        let b = self.brush.update(now);
        let p = self.fill.update(now);
        let layer = self.layer.update(now);

        let r = (b.round() as i32 - 1).max(1);
        let mut k = points_for_fill_fraction(r, p);
        k = ((k as f32) * layer).max(0.0) as usize;

        (x, y, r, k)
    }
}

struct AutoPilot {
    rng: StdRng,

    speed_ips: Transition,

    // colors 0..4
    colors: [ColorTransition; 5],

    // cyclic hue drift (keeps cyclic from “locking”)
    cyclic_hue_offset: Transition,

    reset_at: f32,
    cursors: Vec<AutoCursor>,

    disk_pool: RandomDiskPool,
}

impl AutoPilot {
    fn new(
        gw: usize,
        gh: usize,
        is_screensaver: bool,
        grid_size: u32,
        monitor_rects_px: &[MonRect],
        virtual_origin_px: (i32, i32),
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_1234_5678);

        let speed_ips = Transition::new(&mut rng, 12.0, 5.0, 20.0, 4.0, 10.0);

        let colors = [
            ColorTransition::new(&mut rng, [10, 10, 15], 8.0, 14.0),
            ColorTransition::new(&mut rng, [0, 255, 128], 4.0, 10.0),
            ColorTransition::new(&mut rng, [0, 120, 255], 4.0, 10.0),
            ColorTransition::new(&mut rng, [255, 60, 60], 4.0, 10.0),
            ColorTransition::new(&mut rng, [255, 220, 0], 4.0, 10.0),
        ];

        let cyclic_hue_offset = Transition::new(&mut rng, 0.0, 0.0, 1.0, 4.0, 10.0);

        let max_brush = (gw.min(gh) as i32 / 5).max(10);

        let mut cursors = Vec::new();
        if is_screensaver && cfg!(windows) && !monitor_rects_px.is_empty() {
            let (vx, vy) = virtual_origin_px;
            for m in monitor_rects_px {
                let gl = ((m.l - vx) as f32 / grid_size as f32).floor() as i32;
                let gt = ((m.t - vy) as f32 / grid_size as f32).floor() as i32;
                let gr = ((m.r - vx) as f32 / grid_size as f32).floor() as i32;
                let gb = ((m.b - vy) as f32 / grid_size as f32).floor() as i32;

                let gl = clamp(gl, 0, gw as i32 - 1);
                let gt = clamp(gt, 0, gh as i32 - 1);
                let gr = clamp(gr, 1, gw as i32);
                let gb = clamp(gb, 1, gh as i32);

                let n = rng.random_range(1..=3);
                for _ in 0..n {
                    let sx = rng.random_range(gl as f32..(gr.max(gl + 1) as f32 - 1.0).max(gl as f32));
                    let sy = rng.random_range(gt as f32..(gb.max(gt + 1) as f32 - 1.0).max(gt as f32));
                    cursors.push(AutoCursor::new(&mut rng, gw, gh, max_brush, Some((sx, sy))));
                }
            }
        } else {
            let n = if is_screensaver { rng.random_range(1..=3) } else { 3 };
            for _ in 0..n {
                cursors.push(AutoCursor::new(&mut rng, gw, gh, max_brush, None));
            }
        }

        // pool size: enough to be effectively non-repeating at runtime
        let disk_pool = RandomDiskPool::new(&mut rng, 400_000);

        Self {
            rng,
            speed_ips,
            colors,
            cyclic_hue_offset,
            reset_at: 60.0,
            cursors,
            disk_pool,
        }
    }

    fn pick_color(&mut self) -> [u8; 3] {
        [
            self.rng.random_range(40..=255),
            self.rng.random_range(40..=255),
            self.rng.random_range(40..=255),
        ]
    }

    fn reset(&mut self, now: f32, sim: &mut Sim) {
        // variant
        let all = Variant::all();
        let v = all[self.rng.random_range(0..all.len())];
        sim.variant = v;

        // randomize a few variant params (safe)
        if sim.variant == Variant::Cyclic {
            sim.cyclic_states = self.rng.random_range(8..=18);
        }
        if sim.variant == Variant::GreenbergHastings {
            sim.gh_refrac_max = self.rng.random_range(8..=16);
        }

        // color drift targets (1..4)
        for k in 1..=4 {
            let col = self.pick_color();
            let dur = self.rng.random_range(4.0..10.0);
            self.colors[k].set_target(now, col, dur);
        }
        // background sticks to base (like Python)
        self.colors[0].set_target(now, [10, 10, 15], self.rng.random_range(8.0..14.0));

        // speed nudged
        self.speed_ips.set_target(now, self.rng.random_range(5.0..20.0), self.rng.random_range(4.0..10.0));

        // cyclic hue drift target
        self.cyclic_hue_offset
            .set_target(now, self.rng.random_range(0.0..1.0), self.rng.random_range(4.0..10.0));

        sim.clear();
        sim.seed_random(&mut self.rng);

        self.reset_at = now + 60.0;
    }

    fn tick_frame(&mut self, now: f32, sim: &mut Sim) -> (f32, [[u8; 3]; 5], f32) {
        if now >= self.reset_at {
            self.reset(now, sim);
        }

        self.speed_ips.maybe_new_target(now, &mut self.rng, |r| r.random_range(5.0..20.0));
        let speed = self.speed_ips.update(now);

        // colors 1..4 drift, 0 stays near bg
        for k in 1..=4 {
            self.colors[k].maybe_new_target(now, &mut self.rng, |r| {
                [
                    r.random_range(40..=255),
                    r.random_range(40..=255),
                    r.random_range(40..=255),
                ]
            });
        }
        self.colors[0].maybe_new_target(now, &mut self.rng, |_| [10, 10, 15]);

        self.cyclic_hue_offset
            .maybe_new_target(now, &mut self.rng, |r| r.random_range(0.0..1.0));
        let hue_off = self.cyclic_hue_offset.update(now);

        let mut pal = [[0u8; 3]; 5];
        for i in 0..=4 {
            pal[i] = self.colors[i].update(now);
        }

        (speed, pal, hue_off)
    }

    // Python-like 60 Hz paint tick
    fn tick_paint(&mut self, now: f32, sim: &mut Sim) {
        for c in &mut self.cursors {
            let (x, y, r, k) = c.tick(now, &mut self.rng);
            if k == 0 {
                continue;
            }

            match sim.variant {
                Variant::GrayScott => {
                    // paint acts as V-blob injection
                    sim.seed_blob_gray_scott(x, y, r.max(8));
                }
                _ => {
                    let val = c.pick_value(&mut self.rng, sim.variant, sim);
                    paint_disk_points(sim, &mut self.disk_pool, x, y, r, k, val);
                }
            }
        }
    }

    fn paint_user(&mut self, sim: &mut Sim, gx: i32, gy: i32, brush_size: i32, fill: f32, value: u8) {
        let r = (brush_size - 1).max(1);
        let mut k = points_for_fill_fraction(r, fill);
        if k == 0 {
            k = 1;
        }
        let maxk = sim.w * sim.h;
        if k > maxk {
            k = maxk;
        }

        match sim.variant {
            Variant::GrayScott => {
                if value != 0 {
                    sim.seed_blob_gray_scott(gx, gy, r.max(8));
                }
            }
            _ => {
                paint_disk_points(sim, &mut self.disk_pool, gx, gy, r, k, value);
            }
        }
    }
}

// -----------------------------
// Palette mapping
// -----------------------------
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = (h % 1.0 + 1.0) % 1.0;
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i as i32 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    ]
}

fn build_palette(sim: &Sim, base: &[[u8; 3]; 5], cyclic_hue_offset: f32) -> [[u8; 4]; 256] {
    let mut pal = [[0u8; 4]; 256];

    // Default: map states 0..4 directly to base colors (Python behavior).
    for i in 0..=4 {
        pal[i] = [base[i][0], base[i][1], base[i][2], 255];
    }

    match sim.variant {
        Variant::Cyclic => {
            let k = sim.cyclic_states.max(2) as usize;
            // CRITICAL FIX: keep state 0 as background; do NOT overwrite it.
            pal[0] = [base[0][0], base[0][1], base[0][2], 255];
            for s in 1..k.min(256) {
                let hue = (cyclic_hue_offset + (s as f32 / k as f32)) % 1.0;
                let rgb = hsv_to_rgb(hue, 0.95, 1.0);
                pal[s] = [rgb[0], rgb[1], rgb[2], 255];
            }
        }
        Variant::GreenbergHastings => {
            // 0 bg, 1 excited, 2.. refractory fade
            pal[0] = [base[0][0], base[0][1], base[0][2], 255];
            pal[1] = [base[3][0], base[3][1], base[3][2], 255]; // excited = accent
            let m = sim.gh_refrac_max.max(3) as usize;
            for s in 2..=m.min(255) {
                let t = (s - 2) as f32 / (m.saturating_sub(2).max(1)) as f32;
                let a = base[2]; // start refractory color
                let b = base[0]; // fade to bg
                let rgb = [
                    (a[0] as f32 * (1.0 - t) + b[0] as f32 * t) as u8,
                    (a[1] as f32 * (1.0 - t) + b[1] as f32 * t) as u8,
                    (a[2] as f32 * (1.0 - t) + b[2] as f32 * t) as u8,
                ];
                pal[s] = [rgb[0], rgb[1], rgb[2], 255];
            }
        }
        _ => {}
    }

    pal
}

// -----------------------------
// Rendering (wgpu)
// -----------------------------
struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    blit_pipeline: wgpu::RenderPipeline,
    blit_bind: wgpu::BindGroup,
    blit_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,

    tex: wgpu::Texture,
    tex_view: wgpu::TextureView,

    tex_w: u32,
    tex_h: u32,
    bpr: u32,
    upload: Vec<u8>,
}

impl Gfx {
    async fn new(window: Arc<Window>, surface_w: u32, surface_h: u32, tex_w: u32, tex_h: u32) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapters found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::default(),
                },
            )
            .await
            .expect("request_device failed");

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes[0]
        };
        let alpha_mode = caps.alpha_modes[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: surface_w.max(1),
            height: surface_h.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (tex, tex_view, tex_w, tex_h, bpr, upload) = Self::make_pixel_texture(&device, tex_w, tex_h);

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });

        let blit_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bind"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit_shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl_layout"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            blit_pipeline,
            blit_bind,
            blit_bgl,
            blit_sampler,
            tex,
            tex_view,
            tex_w,
            tex_h,
            bpr,
            upload,
        }
    }

    fn make_pixel_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView, u32, u32, u32, Vec<u8>) {
        let tex_w = w.max(1);
        let tex_h = h.max(1);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pixel_tex"),
            size: wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let tight_bpr = 4 * tex_w;
        let bpr = ((tight_bpr + 255) / 256) * 256;
        let upload = vec![0u8; (bpr * tex_h) as usize];

        (tex, tex_view, tex_w, tex_h, bpr, upload)
    }

    fn resize(&mut self, surface_w: u32, surface_h: u32, tex_w: u32, tex_h: u32) {
        self.config.width = surface_w.max(1);
        self.config.height = surface_h.max(1);
        self.surface.configure(&self.device, &self.config);

        let (tex, tex_view, tw, th, bpr, upload) = Self::make_pixel_texture(&self.device, tex_w, tex_h);
        self.tex = tex;
        self.tex_view = tex_view;
        self.tex_w = tw;
        self.tex_h = th;
        self.bpr = bpr;
        self.upload = upload;

        self.blit_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bind"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.tex_view),
                },
            ],
        });
    }

    fn upload_pixels(&mut self, sim: &Sim, palette: &[[u8; 4]; 256]) {
        let w = sim.w.min(self.tex_w as usize);
        let h = sim.h.min(self.tex_h as usize);
        let bpr = self.bpr as usize;

        self.upload.par_chunks_mut(bpr).enumerate().take(h).for_each(|(y, row)| {
            let base = y * sim.w;
            let mut off = 0usize;
            for x in 0..w {
                let v = sim.cur[base + x] as usize;
                let c = palette[v.min(255)];
                row[off] = c[0];
                row[off + 1] = c[1];
                row[off + 2] = c[2];
                row[off + 3] = c[3];
                off += 4;
            }
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.bpr),
                rows_per_image: Some(self.tex_h),
            },
            wgpu::Extent3d {
                width: self.tex_w,
                height: self.tex_h,
                depth_or_array_layers: 1,
            },
        );
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("enc"),
        });

        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.blit_pipeline);
            rp.set_bind_group(0, &self.blit_bind, &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(Some(enc.finish()));
        frame.present();
    }
}

// -----------------------------
// App
// -----------------------------
struct App {
    is_screensaver: bool,
    debug: bool,

    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,

    sim: Option<Sim>,
    autopilot: Option<AutoPilot>,

    input: InputState,

    grid_size: u32,

    // user controls (Python-like)
    user_brush_size: i32,
    user_fill: f32,
    user_palette_sel: u8,

    // timing
    t0: Instant,
    last_frame: Instant,

    sim_accum: f32,
    logic_accum: f32,

    // debug stats
    last_stat: Instant,
    frames: u64,
    sim_steps: u64,
    sys: System,
}

impl App {
    fn new(is_screensaver: bool, debug: bool) -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        sys.refresh_all();

        Self {
            is_screensaver,
            debug,
            window: None,
            gfx: None,
            sim: None,
            autopilot: None,
            input: InputState::default(),
            grid_size: 1,

            user_brush_size: 24,
            user_fill: 0.010,
            user_palette_sel: 1,

            t0: Instant::now(),
            last_frame: Instant::now(),
            sim_accum: 0.0,
            logic_accum: 0.0,

            last_stat: Instant::now(),
            frames: 0,
            sim_steps: 0,
            sys,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        #[cfg(windows)]
        let (vx, vy, vw, vh, rects) = virtual_bounds_windows();

        #[cfg(not(windows))]
        let (vx, vy, vw, vh, rects) = (0i32, 0i32, 1280i32, 720i32, Vec::<MonRect>::new());

        // IMPORTANT: grid_size computed from virtual desktop short edge (Python).
        self.grid_size = choose_grid_size_explosive(vw, vh);

        let (win_w, win_h, pos) = if self.is_screensaver {
            (vw as u32, vh as u32, Some((vx, vy)))
        } else {
            (1280u32, 720u32, None)
        };

        let mut attrs = WindowAttributes::default()
            .with_title("Conway Screensaver")
            .with_decorations(!self.is_screensaver)
            .with_resizable(!self.is_screensaver)
            .with_inner_size(PhysicalSize::new(win_w, win_h));

        if let Some((px, py)) = pos {
            attrs = attrs.with_position(PhysicalPosition::new(px, py));
        }

        let win = Arc::new(el.create_window(attrs).expect("create_window"));
        win.set_cursor_visible(true);
        let _ = win.set_cursor_grab(CursorGrabMode::None);

        let size = win.inner_size();

        let gw = (size.width / self.grid_size.max(1)).max(1) as usize;
        let gh = (size.height / self.grid_size.max(1)).max(1) as usize;

        let mut sim = Sim::new(gw, gh);

        let mut ap = AutoPilot::new(
            gw,
            gh,
            self.is_screensaver,
            self.grid_size,
            &rects,
            (vx, vy),
        );
        ap.reset(0.0, &mut sim);

        let gfx = block_on(Gfx::new(win.clone(), size.width, size.height, gw as u32, gh as u32));

        self.window = Some(win);
        self.gfx = Some(gfx);
        self.sim = Some(sim);
        self.autopilot = Some(ap);

        self.t0 = Instant::now();
        self.last_frame = Instant::now();
        self.sim_accum = 0.0;
        self.logic_accum = 0.0;

        self.last_stat = Instant::now();
        self.frames = 0;
        self.sim_steps = 0;
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    use winit::keyboard::{Key, NamedKey};
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => el.exit(),
                        Key::Character(c) => {
                            // palette select like Python (1..4)
                            if c == "1" { self.user_palette_sel = 1; }
                            if c == "2" { self.user_palette_sel = 2; }
                            if c == "3" { self.user_palette_sel = 3; }
                            if c == "4" { self.user_palette_sel = 4; }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(sz) => {
                if let (Some(gfx), Some(sim), Some(_ap)) = (&mut self.gfx, &mut self.sim, &mut self.autopilot) {
                    let gw = (sz.width / self.grid_size.max(1)).max(1) as usize;
                    let gh = (sz.height / self.grid_size.max(1)).max(1) as usize;

                    if !self.is_screensaver && (gw != sim.w || gh != sim.h) {
                        *sim = Sim::new(gw, gh);
                        // keep autopilot; it will reset on next frame anyway
                    }
                    gfx.resize(sz.width, sz.height, gw as u32, gh as u32);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_px = Some((position.x as f32, position.y as f32));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.input.left_down = down,
                    MouseButton::Right => self.input.right_down = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.wheel_delta += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        el.set_control_flow(ControlFlow::Poll);

        let now_i = Instant::now();
        let mut dt = (now_i - self.last_frame).as_secs_f32();
        self.last_frame = now_i;
        if dt < 0.0 { dt = 0.0; }
        if dt > 0.25 { dt = 0.25; }

        let (sim, ap) = match (&mut self.sim, &mut self.autopilot) {
            (Some(s), Some(a)) => (s, a),
            _ => return,
        };

        let now = (now_i - self.t0).as_secs_f32();

        // Frame tick (speed + palette + reset)
        let (speed_ips, base_pal, cyclic_hue_offset) = ap.tick_frame(now, sim);

        // Mouse wheel = brush size (Python-ish)
        if self.input.wheel_delta.abs() > 0.001 {
            let max_user = (sim.w.min(sim.h) as i32 / 3).max(10);
            self.user_brush_size = clamp(self.user_brush_size + (self.input.wheel_delta.round() as i32), 1, max_user);
            self.input.wheel_delta = 0.0;
        }

        // 60Hz paint tick (autocursors)
        self.logic_accum += dt;
        let logic_step = 1.0 / 60.0;
        while self.logic_accum >= logic_step {
            self.logic_accum -= logic_step;
            ap.tick_paint(now, sim);
        }

        // One user paint call per frame (Python)
        if self.input.left_down || self.input.right_down {
            if let Some((mx, my)) = self.input.cursor_px {
                let gx = (mx / self.grid_size as f32).floor() as i32;
                let gy = (my / self.grid_size as f32).floor() as i32;
                let val = if self.input.left_down { self.user_palette_sel } else { 0 };
                ap.paint_user(sim, gx, gy, self.user_brush_size, self.user_fill, val);
            }
        }

        // Simulation steps (speed_ips)
        self.sim_accum += dt;
        let step_interval = 1.0 / speed_ips.max(1e-6);
        let dt_gs = step_interval.min(0.02);
        let mut steps = 0usize;
        let max_steps = 32; // Python uses 32 cap

        while self.sim_accum >= step_interval && steps < max_steps {
            self.sim_accum -= step_interval;
            sim.step(dt_gs);
            steps += 1;
        }
        self.sim_steps += steps as u64;

        // Render
        if let Some(gfx) = &mut self.gfx {
            let pal = build_palette(sim, &base_pal, cyclic_hue_offset);
            gfx.upload_pixels(sim, &pal);
            gfx.render();
        }

        // Debug prints
        self.frames += 1;
        if self.debug && self.last_stat.elapsed() >= Duration::from_secs(1) {
            self.sys.refresh_cpu_all();
            self.sys.refresh_memory();

            let cpu = self.sys.global_cpu_usage();
            let mem_used = self.sys.used_memory();
            let mem_total = self.sys.total_memory();
            let mem_pct = if mem_total > 0 { (mem_used as f32 / mem_total as f32) * 100.0 } else { 0.0 };

            eprintln!(
                "[DEBUG] CPU {:5.1}% | MEM {:5.1}% | FPS {} | steps/s {} | variant {:?} | speed_ips {:.2} | grid {} | brush {}",
                cpu,
                mem_pct,
                self.frames,
                self.sim_steps,
                sim.variant,
                speed_ips,
                self.grid_size,
                self.user_brush_size
            );

            self.frames = 0;
            self.sim_steps = 0;
            self.last_stat = Instant::now();
        }
    }
}

// -----------------------------
// CLI parsing
// -----------------------------
fn parse_args() -> (bool, bool) {
    let mut is_screensaver = false;
    let mut debug = false;

    for a in std::env::args().skip(1) {
        let a = a.to_lowercase();
        if a == "--debug" {
            debug = true;
        }
        if a == "/s" || a == "--screensaver" {
            is_screensaver = true;
        }
        if a.starts_with("/c") || a.starts_with("/p") {
            std::process::exit(0);
        }
    }

    (is_screensaver, debug)
}

fn main() -> Result<(), winit::error::EventLoopError> {
    let (is_screensaver, debug) = parse_args();
    let event_loop = EventLoop::new()?;
    let mut app = App::new(is_screensaver, debug);
    event_loop.run_app(&mut app)
}
