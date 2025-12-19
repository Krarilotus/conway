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
    keyboard::{Key, NamedKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

#[cfg(windows)]
use windows_sys::Win32::{
    Foundation::{LPARAM, RECT},
    Graphics::Gdi::{EnumDisplayMonitors, GetMonitorInfoW, HDC, HMONITOR, MONITORINFO},
};

mod tuning;


// -----------------------------
// Embedded WGSL shaders (no external files)
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

#[derive(Default, Clone, Copy)]
struct InputState {
    cursor: Option<(f32, f32)>,
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
    primary: bool,
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
                let primary = (mi.dwFlags & 1) != 0; // MONITORINFOF_PRIMARY
                out.push(MonRect {
                    l: r.left,
                    t: r.top,
                    r: r.right,
                    b: r.bottom,
                    primary,
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

// -----------------------------
// Transition helpers
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
    fn new(v: f32, lo: f32, hi: f32, min_dur: f32, max_dur: f32) -> Self {
        let now = 0.0;
        Self {
            cur: v,
            start: v,
            target: v,
            t0: now,
            dur: 1.0,
            next_change: now + min_dur,
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

    fn maybe_new_target<R: Rng + ?Sized, F: FnOnce(&mut R) -> f32>(
        &mut self,
        now: f32,
        rng: &mut R,
        pick: F,
    ) {
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
    fn new(rgb: [u8; 3], min_dur: f32, max_dur: f32) -> Self {
        Self {
            cur: rgb,
            start: rgb,
            target: rgb,
            t0: 0.0,
            dur: 1.0,
            next_change: min_dur,
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

    fn maybe_new_target<R: Rng + ?Sized, F: FnOnce(&mut R) -> [u8; 3]>(
        &mut self,
        now: f32,
        rng: &mut R,
        pick: F,
    ) {
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
// Variants
// -----------------------------
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Variant {
    Standard,
    HighLife,
    DayNight,
    Immigration,
    QuadLife,
    Generations,
    BriansBrain,
    LargerThanLife,
    Margolus,
    Cyclic,
    GrayScott,
}

impl Variant {
    fn all() -> &'static [Variant] {
        &[
            Variant::Standard,
            Variant::HighLife,
            Variant::DayNight,
            Variant::Immigration,
            Variant::QuadLife,
            Variant::Generations,
            Variant::BriansBrain,
            Variant::LargerThanLife,
            Variant::Margolus,
            Variant::Cyclic,
            Variant::GrayScott,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Variant::Standard => "STANDARD",
            Variant::HighLife => "HIGHLIFE",
            Variant::DayNight => "DAY-NIGHT",
            Variant::Immigration => "IMMIGRATION",
            Variant::QuadLife => "QUADLIFE",
            Variant::Generations => "GENERATIONS",
            Variant::BriansBrain => "BRIAN'S BRAIN",
            Variant::LargerThanLife => "LARGER THAN LIFE",
            Variant::Margolus => "MARGOLUS",
            Variant::Cyclic => "CYCLIC",
            Variant::GrayScott => "GRAY-SCOTT",
        }
    }
}

// -----------------------------
// Simulation
// -----------------------------
struct Sim {
    w: usize,
    h: usize,
    n: usize,

    variant: Variant,
    parity: u8,

    cur: Vec<u8>,
    next: Vec<u8>,

    // Rolling sums scratch
    hsum: Vec<u16>,
    vsum: Vec<u16>,

    // Scratch buffers to avoid Vec::clone in hot loops
    tmp1: Vec<u16>,
    tmp2: Vec<u16>,
    tmp3: Vec<u16>,
    tmp4: Vec<u16>,
    tmp5: Vec<u16>,

    // Gray-Scott fields
    gs_u: Vec<f32>,
    gs_v: Vec<f32>,
    gs_u2: Vec<f32>,
    gs_v2: Vec<f32>,

    // Cyclic
    cyclic_states: u8,

    // LTL parameters
    ltl_r: usize,
    ltl_birth_lo: f32,
    ltl_birth_hi: f32,
    ltl_surv_lo: f32,
    ltl_surv_hi: f32,

    // Generations / trails
    gen_decay_max: u8,
    bb_refractory_max: u8,
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

            gs_u: vec![1.0; n],
            gs_v: vec![0.0; n],
            gs_u2: vec![1.0; n],
            gs_v2: vec![0.0; n],

            cyclic_states: 12,

            ltl_r: 5,
            ltl_birth_lo: 0.22,
            ltl_birth_hi: 0.34,
            ltl_surv_lo: 0.18,
            ltl_surv_hi: 0.48,

            gen_decay_max: 8,
            bb_refractory_max: 5,
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
    }

    fn seed_random<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        crate::tuning::seed_sim(self, rng, crate::tuning::VariantTuning::default());
    }

    /// Scatter exactly `k` random points inside a disk (cx, cy, radius).
    /// This matches the "fill fraction -> k points" semantics used by the autopilot.
    fn scatter_disk_points<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        cx: i32,
        cy: i32,
        radius: i32,
        k: usize,
        value: u8,
    ) {
        if k == 0 {
            return;
        }

        let w = self.w as i32;
        let h = self.h as i32;

        let r = radius.max(1);
        let r2 = r * r;

        // Rejection sampling inside a square bounding box.
        // For your typical k this is totally fine and keeps semantics simple.
        for _ in 0..k {
            let mut dx;
            let mut dy;
            loop {
                dx = rng.random_range(-r..=r);
                dy = rng.random_range(-r..=r);
                if dx * dx + dy * dy <= r2 {
                    break;
                }
            }

            let x = (cx + dx).rem_euclid(w) as usize;
            let y = (cy + dy).rem_euclid(h) as usize;
            let i = y * self.w + x;
            self.cur[i] = value;
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

    fn paint_disk<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        cx: i32,
        cy: i32,
        radius: i32,
        value: u8,
        density: f32,
    ) {
        let w = self.w as i32;
        let h = self.h as i32;
        let r2 = radius * radius;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                if rng.random::<f32>() >= density {
                    continue;
                }
                let x = (cx + dx).rem_euclid(w) as usize;
                let y = (cy + dy).rem_euclid(h) as usize;
                let i = y * self.w + x;
                self.cur[i] = value;
            }
        }
    }

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
                    let v = cur[base + x];
                    s += mode.to01(v) as u16;
                }
                hrow[0] = s;

                for x in 1..w {
                    let x_add = ((x as i32 + r).rem_euclid(w as i32)) as usize;
                    let x_sub = ((x as i32 - r - 1).rem_euclid(w as i32)) as usize;
                    let v_add = cur[base + x_add];
                    let v_sub = cur[base + x_sub];
                    s += mode.to01(v_add) as u16;
                    s -= mode.to01(v_sub) as u16;
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
                let sum3 = self.vsum[i] as i32;
                let n = sum3 - (alive as i32);
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
                    if b == mx {
                        out = 2;
                    }
                    if c == mx {
                        out = 3;
                    }
                    if d == mx {
                        out = 4;
                    }
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
                let sum3 = self.vsum[i] as i32;
                let n = sum3 - (any as i32);
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

    fn step_brians_brain(&mut self) {
        self.moore_sum_mask(1, MaskMode::Eq(1));
        let w = self.w;
        let cur = &self.cur;
        let next = &mut self.next;
        let maxr = self.bb_refractory_max;

        next.par_chunks_mut(w).enumerate().for_each(|(y, nrow)| {
            let y = y as usize;
            for x in 0..w {
                let i = y * w + x;
                let v = cur[i];
                if v == 0 {
                    let n = self.vsum[i] as i32;
                    nrow[x] = if n == 2 { 1 } else { 0 };
                } else if v == 1 {
                    nrow[x] = 2;
                } else {
                    let nv = v.saturating_add(1);
                    nrow[x] = if nv > maxr { 0 } else { nv };
                }
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
        let k = self.cyclic_states.max(2);
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
                let want = ((v as u16 + 1) % k as u16) as u8;

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

    fn step_larger_than_life(&mut self) {
        let r = self.ltl_r as i32;
        self.moore_sum_mask(r, MaskMode::Alive);

        let area = ((2 * r + 1) * (2 * r + 1) - 1) as i32;

        let b_lo = (self.ltl_birth_lo * area as f32).round() as i32;
        let b_hi = (self.ltl_birth_hi * area as f32).round() as i32;
        let s_lo = (self.ltl_surv_lo * area as f32).round() as i32;
        let s_hi = (self.ltl_surv_hi * area as f32).round() as i32;

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

                let born = !alive && (n >= b_lo && n <= b_hi);
                let surv = alive && (n >= s_lo && n <= s_hi);
                nrow[x] = if born || surv { 1 } else { 0 };
            }
        });

        std::mem::swap(&mut self.cur, &mut self.next);
    }

    fn step_gray_scott(&mut self, dt: f32) {
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
            Variant::Standard => self.step_life_like(LifeRule::standard()),
            Variant::HighLife => self.step_life_like(LifeRule::highlife()),
            Variant::DayNight => self.step_life_like(LifeRule::daynight()),
            Variant::Immigration => self.step_immigration(),
            Variant::QuadLife => self.step_quadlife(),
            Variant::Generations => self.step_generations(),
            Variant::BriansBrain => self.step_brians_brain(),
            Variant::LargerThanLife => self.step_larger_than_life(),
            Variant::Margolus => self.step_margolus(),
            Variant::Cyclic => self.step_cyclic(),
            Variant::GrayScott => self.step_gray_scott(dt_for_gs),
        }
    }
}

#[derive(Copy, Clone)]
struct LifeRule {
    survive_mask: u16,
    birth_mask: u16,
}
impl LifeRule {
    fn standard() -> Self {
        Self {
            survive_mask: (1 << 2) | (1 << 3),
            birth_mask: (1 << 3),
        }
    }
    fn highlife() -> Self {
        Self {
            survive_mask: (1 << 2) | (1 << 3),
            birth_mask: (1 << 3) | (1 << 6),
        }
    }
    fn daynight() -> Self {
        let mut s = 0u16;
        for &n in &[3, 4, 6, 7, 8] {
            s |= 1 << n;
        }
        let mut b = 0u16;
        for &n in &[3, 6, 7, 8] {
            b |= 1 << n;
        }
        Self {
            survive_mask: s,
            birth_mask: b,
        }
    }
    fn survive(&self, n: i32) -> bool {
        if n < 0 || n > 15 {
            return false;
        }
        (self.survive_mask & (1 << n)) != 0
    }
    fn birth(&self, n: i32) -> bool {
        if n < 0 || n > 15 {
            return false;
        }
        (self.birth_mask & (1 << n)) != 0
    }
}

#[derive(Copy, Clone)]
enum MaskMode {
    Alive,
    Eq(u8),
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

// -----------------------------
// Autopilot (fully driven by tuning::ControlPanel)
// -----------------------------
struct AutoCursor {
    // position
    x: Transition,
    y: Transition,

    // 60Hz stamp parameters
    brush: Transition,
    fill: Transition,
    layer: Transition,
}

impl AutoCursor {
    fn new<R: Rng + ?Sized>(rng: &mut R, w: usize, h: usize, panel: &crate::tuning::ControlPanel) -> Self {
        let w1 = (w.max(1) as f32 - 1.0).max(0.0);
        let h1 = (h.max(1) as f32 - 1.0).max(0.0);

        let x0 = rng.random_range(0.0..w1.max(1.0));
        let y0 = rng.random_range(0.0..h1.max(1.0));

        // Cursor motion retarget windows
        let xy_rt = panel.cursors.motion.xy_retarget;
        let mut x = Transition::new(x0, 0.0, w1, xy_rt.min, xy_rt.max);
        let mut y = Transition::new(y0, 0.0, h1, xy_rt.min, xy_rt.max);

        // Match your old semantics: start stationary; first change happens after a random duration.
        x.set_target(0.0, x0, xy_rt.pick(rng));
        y.set_target(0.0, y0, xy_rt.pick(rng));

        // Stamp knobs (brush, fill fraction, layer multiplier)
        let b = panel.cursors.stamp_60hz.brush_radius_px;
        let f = panel.cursors.stamp_60hz.fill_fraction;
        let l = panel.cursors.stamp_60hz.layer_multiplier;

        // Use make_transition() helper
        let mut brush = b.make_transition();
        let mut fill  = f.make_transition();
        let mut layer = l.make_transition();

        // Desync “next change time” per cursor without changing value yet.
        brush.set_target(0.0, brush.cur, b.retarget.pick(rng));
        fill.set_target(0.0, fill.cur, f.retarget.pick(rng));
        layer.set_target(0.0, layer.cur, l.retarget.pick(rng));

        Self { x, y, brush, fill, layer }
    }
}

struct AutoPilot {
    rng: StdRng,
    panel: crate::tuning::ControlPanel,

    // global “time speed”
    speed_ips: Transition,

    // blob injector (step-based)
    blob_rate: Transition,    // blobs per second
    blob_radius: Transition,  // pixels
    blob_density: Transition, // per-cell probability inside blob disk

    // step-based cursor injector
    step_cursor_rate: Transition,    // stamps per second
    step_cursor_brush: Transition,   // pixels
    step_cursor_density: Transition, // per-cell probability in stamp disk

    // Larger-than-life drifting params (fed into Sim each frame)
    ltl_r: Transition,
    ltl_birth_lo: Transition,
    ltl_birth_hi: Transition,
    ltl_surv_lo: Transition,
    ltl_surv_hi: Transition,

    // palette (base palette sent into build_palette)
    pal0: [ColorTransition; 5],

    reset_at: f32,
    num_cursors: usize,
    cursors: Vec<AutoCursor>,
}

impl AutoPilot {
    /// Convert a “fill fraction” (0..1) into a number of points to scatter in a disk.
    /// Intuition: higher fill => more points.
    fn points_for_fill_fraction(radius: i32, fill: f32) -> usize {
        let r = radius.max(1) as f32;
        let p = fill.clamp(0.0, 0.999_999);
        if p <= 0.0 {
            return 0;
        }
        let area = std::f32::consts::PI * r * r;
        let k = (-area * (1.0 - p).ln()).round() as i32;
        k.max(0) as usize
    }

    fn new(w: usize, h: usize, num_cursors: usize) -> Self {
        let panel = crate::tuning::ControlPanel::for_sim(w, h);
        let mut rng = StdRng::seed_from_u64(panel.rng_seed);

        // Build all animated transitions using helpers
        let speed_ips = panel.sim_steps_per_second.make_transition();

        let blob_rate = panel.blobs.blobs_per_second.make_transition();
        let blob_radius = panel.blobs.blob_radius_px.make_transition();
        let blob_density = panel.blobs.blob_cell_probability.make_transition();

        let step_cursor_rate = panel.step_cursor.stamps_per_second.make_transition();
        let step_cursor_brush = panel.step_cursor.brush_radius_px.make_transition();
        let step_cursor_density = panel.step_cursor.stamp_cell_probability.make_transition();

        let ltl_r = panel.ltl.radius_r.make_transition();
        let ltl_birth_lo = panel.ltl.birth_lo.make_transition();
        let ltl_birth_hi = panel.ltl.birth_hi.make_transition();
        let ltl_surv_lo  = panel.ltl.survive_lo.make_transition();
        let ltl_surv_hi  = panel.ltl.survive_hi.make_transition();

        // Palette transitions from the panel
        let mut pal0 = [
            panel.palette.background.make_transition(),
            panel.palette.accent.make_transition(),
            panel.palette.accent.make_transition(),
            panel.palette.accent.make_transition(),
            panel.palette.accent.make_transition(),
        ];

        // At startup, optionally randomize accent targets immediately
        if panel.reset_policy.randomize_palette_on_reset {
            let acc = panel.palette.accent;
            for k in 1..=4 {
                let dur = acc.retarget.pick(&mut rng);
                let rgb = acc.pick_random_rgb(&mut rng);
                pal0[k].set_target(0.0, rgb, dur);
            }
            if panel.palette.background.enabled {
                let bg = panel.palette.background;
                let dur = bg.retarget.pick(&mut rng);
                let rgb = bg.pick_random_rgb(&mut rng);
                pal0[0].set_target(0.0, rgb, dur);
            }
        }

        // Cursors
        let nc = num_cursors.max(1);
        let mut cursors = Vec::with_capacity(nc);
        for _ in 0..nc {
            cursors.push(AutoCursor::new(&mut rng, w, h, &panel));
        }

        Self {
            rng,
            panel,
            speed_ips,
            blob_rate,
            blob_radius,
            blob_density,
            step_cursor_rate,
            step_cursor_brush,
            step_cursor_density,
            ltl_r,
            ltl_birth_lo,
            ltl_birth_hi,
            ltl_surv_lo,
            ltl_surv_hi,
            pal0,
            reset_at: 0.0 + 60.0,
            num_cursors: nc,
            cursors,
        }
    }

    fn reset(&mut self, now: f32, sim: &mut Sim) {
        let rp = self.panel.reset_policy;

        // Variant choice
        if now == 0.0 {
            sim.variant = Variant::Standard;
        } else if rp.change_variant {
            let all = Variant::all();
            sim.variant = all[self.rng.random_range(0..all.len())];
        }

        // Randomize per-variant params if desired
        if rp.randomize_variant_params {
            crate::tuning::randomize_variant_params(sim, &mut self.rng, self.panel.variants);
        }

        // Palette reset behavior
        if rp.randomize_palette_on_reset {
            let acc = self.panel.palette.accent;
            for k in 1..=4 {
                let dur = acc.retarget.pick(&mut self.rng);
                let rgb = acc.pick_random_rgb(&mut self.rng);
                self.pal0[k].set_target(now, rgb, dur);
            }
            if self.panel.palette.background.enabled {
                let bg = self.panel.palette.background;
                let dur = bg.retarget.pick(&mut self.rng);
                let rgb = bg.pick_random_rgb(&mut self.rng);
                self.pal0[0].set_target(now, rgb, dur);
            }
        }

        // Cursor reset behavior
        if rp.recenter_cursors_on_reset {
            self.cursors.clear();
            for _ in 0..self.num_cursors {
                self.cursors.push(AutoCursor::new(&mut self.rng, sim.w, sim.h, &self.panel));
            }
        }

        // Sim reseed behavior
        if rp.reseed_sim_state {
            sim.clear();
            crate::tuning::seed_sim(sim, &mut self.rng, self.panel.variants);
        }

        self.reset_at = now + self.panel.reset_after_seconds;
    }

    /// Called once per frame: updates all retargeting knobs + palette, and applies LTL params into the sim.
    fn tick_frame(&mut self, now: f32, sim: &mut Sim) -> (f32, [[u8; 3]; 5]) {
        // --- retarget global knobs using maybe_retarget() helper ---
        self.panel.sim_steps_per_second.maybe_retarget(&mut self.speed_ips, now, &mut self.rng);
        
        self.panel.blobs.blobs_per_second.maybe_retarget(&mut self.blob_rate, now, &mut self.rng);
        self.panel.blobs.blob_radius_px.maybe_retarget(&mut self.blob_radius, now, &mut self.rng);
        self.panel.blobs.blob_cell_probability.maybe_retarget(&mut self.blob_density, now, &mut self.rng);

        self.panel.step_cursor.stamps_per_second.maybe_retarget(&mut self.step_cursor_rate, now, &mut self.rng);
        self.panel.step_cursor.brush_radius_px.maybe_retarget(&mut self.step_cursor_brush, now, &mut self.rng);
        self.panel.step_cursor.stamp_cell_probability.maybe_retarget(&mut self.step_cursor_density, now, &mut self.rng);

        // LTL params drift
        let ltl = self.panel.ltl;
        ltl.radius_r.maybe_retarget(&mut self.ltl_r, now, &mut self.rng);
        ltl.birth_lo.maybe_retarget(&mut self.ltl_birth_lo, now, &mut self.rng);
        ltl.birth_hi.maybe_retarget(&mut self.ltl_birth_hi, now, &mut self.rng);
        ltl.survive_lo.maybe_retarget(&mut self.ltl_surv_lo, now, &mut self.rng);
        ltl.survive_hi.maybe_retarget(&mut self.ltl_surv_hi, now, &mut self.rng);

        // Palette drift (independent knobs)
        let bg = self.panel.palette.background;
        bg.maybe_retarget(&mut self.pal0[0], now, &mut self.rng);

        let acc = self.panel.palette.accent;
        for k in 1..=4 {
            acc.maybe_retarget(&mut self.pal0[k], now, &mut self.rng);
        }

        // --- update transitions (advance easing) ---
        let speed = self.speed_ips.update(now);

        // Apply LTL into sim (continuous drift)
        sim.ltl_r = self.ltl_r.update(now).round().clamp(2.0, 7.0) as usize;
        sim.ltl_birth_lo = self.ltl_birth_lo.update(now);
        sim.ltl_birth_hi = self.ltl_birth_hi.update(now);
        sim.ltl_surv_lo = self.ltl_surv_lo.update(now);
        sim.ltl_surv_hi = self.ltl_surv_hi.update(now);

        // Compute palette output
        let mut pal = [[0u8; 3]; 5];
        for i in 0..=4 {
            pal[i] = self.pal0[i].update(now);
        }

        // Reset if needed
        if now >= self.reset_at {
            self.reset(now, sim);
        }

        (speed, pal)
    }

    /// Called at sim-step frequency: injects blobs + step-based stamps.
    fn tick_step(&mut self, speed_ips: f32, sim: &mut Sim) {
        // --- blob injection: blobs_per_second converted into per-step probability ---
        let blobs_per_sec = self.blob_rate.cur;
        let p_blob = (blobs_per_sec / speed_ips.max(1e-6)).clamp(0.0, 1.0);

        if self.rng.random::<f32>() < p_blob {
            let cx = self.rng.random_range(0..sim.w) as i32;
            let cy = self.rng.random_range(0..sim.h) as i32;
            let rad = self.blob_radius.cur.round() as i32;
            let dens = self.blob_density.cur;

            let val = crate::tuning::pick_paint_value(&mut self.rng, sim, self.panel.paint_values);

            if sim.variant == Variant::GrayScott {
                sim.seed_blob_gray_scott(cx, cy, rad.max(8));
            } else {
                sim.paint_disk(&mut self.rng, cx, cy, rad, val, dens);
            }
        }

        // --- step-based cursor stamping: stamps_per_second converted into per-step accumulator ---
        let stamps_per_sec = self.step_cursor_rate.cur;
        let add = stamps_per_sec / speed_ips.max(1e-6);

        // One shared stamp brush for this path
        let brush = self.step_cursor_brush.cur.round() as i32;
        let dens = self.step_cursor_density.cur;

        // We reuse the same cursor list, using their current x/y for stamp locations.
        for c in &mut self.cursors {
            // we don't store accum here anymore; just do probabilistic stamping based on add.
            // Equivalent expectation to your old accum loop.
            if self.rng.random::<f32>() < add.clamp(0.0, 1.0) {
                let x = c.x.cur.round() as i32;
                let y = c.y.cur.round() as i32;

                let val = crate::tuning::pick_paint_value(&mut self.rng, sim, self.panel.paint_values);

                if sim.variant == Variant::GrayScott {
                    sim.seed_blob_gray_scott(x, y, brush.max(8));
                } else {
                    sim.paint_disk(&mut self.rng, x, y, brush, val, dens);
                }
            }
        }
    }

    /// Called once per frame: updates cursor x/y/brush motion (NOT fill/layer).
    fn tick_cursor_motion(&mut self, now: f32, sim: &Sim) {
        let w = sim.w;
        let h = sim.h;

        let max_x = (w as f32 - 1.0).max(1.0);
        let max_y = (h as f32 - 1.0).max(1.0);

        for c in &mut self.cursors {
            c.x.maybe_new_target(now, &mut self.rng, |r| r.random_range(0.0..max_x));
            c.y.maybe_new_target(now, &mut self.rng, |r| r.random_range(0.0..max_y));

            // brush retarget is controlled by the stamp brush knob
            let b = self.panel.cursors.stamp_60hz.brush_radius_px;
            if b.enabled {
                c.brush.maybe_new_target(now, &mut self.rng, |r| b.range.pick(r));
            }

            c.x.update(now);
            c.y.update(now);
            c.brush.update(now);
        }
    }

    /// Called at 60Hz: does ONE stamp per cursor per tick (python-like).
    fn tick_paint_60hz(&mut self, now: f32, sim: &mut Sim) {
        let f = self.panel.cursors.stamp_60hz.fill_fraction;
        let l = self.panel.cursors.stamp_60hz.layer_multiplier;

        for c in &mut self.cursors {
            if f.enabled {
                c.fill.maybe_new_target(now, &mut self.rng, |r| f.range.pick(r));
            }
            if l.enabled {
                c.layer.maybe_new_target(now, &mut self.rng, |r| l.range.pick(r));
            }

            // update current values
            let x = c.x.cur.round() as i32;
            let y = c.y.cur.round() as i32;
            let brush = c.brush.cur.round() as i32;

            let fill = c.fill.update(now);
            let layer = c.layer.update(now);

            let r = (brush - 1).max(1);
            let mut k = Self::points_for_fill_fraction(r, fill);
            k = ((k as f32) * layer).round().max(0.0) as usize;

            if k == 0 {
                continue;
            }

            let val = crate::tuning::pick_paint_value(&mut self.rng, sim, self.panel.paint_values);

            if sim.variant == Variant::GrayScott {
                sim.seed_blob_gray_scott(x, y, r.max(8));
            } else {
                sim.scatter_disk_points(&mut self.rng, x, y, r, k, val);
            }
        }
    }

    // Added rand_color helper to fix compiler error
    fn rand_color(&mut self) -> [u8; 3] {
        [
            self.rng.random(),
            self.rng.random(),
            self.rng.random(),
        ]
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

fn build_palette(sim: &Sim, base: &[[u8; 3]; 5]) -> [[u8; 4]; 256] {
    let mut pal = [[0u8; 4]; 256];
    for i in 0..=4 {
        pal[i] = [base[i][0], base[i][1], base[i][2], 255];
    }

    match sim.variant {
        Variant::BriansBrain => {
            pal[0] = [10, 10, 15, 255];
            pal[1] = [255, 255, 255, 255];
            let maxr = sim.bb_refractory_max.max(3);
            for s in 2..=maxr {
                let t = (s - 2) as f32 / (maxr - 2) as f32;
                let c = [
                    0u8,
                    (180.0 + (30.0 - 180.0) * t).round().clamp(0.0, 255.0) as u8,
                    (255.0 + (60.0 - 255.0) * t).round().clamp(0.0, 255.0) as u8,
                ];
                pal[s as usize] = [c[0], c[1], c[2], 255];
            }
        }
        Variant::Generations => {
            pal[0] = [10, 10, 15, 255];
            pal[1] = [base[1][0], base[1][1], base[1][2], 255];
            let m = sim.gen_decay_max.max(4);
            for s in 2..=m {
                let t = (s - 2) as f32 / (m - 2) as f32;
                let c = [
                    (base[1][0] as f32 * (1.0 - t) + 20.0 * t).round() as u8,
                    (base[1][1] as f32 * (1.0 - t) + 20.0 * t).round() as u8,
                    (base[1][2] as f32 * (1.0 - t) + 20.0 * t).round() as u8,
                ];
                pal[s as usize] = [c[0], c[1], c[2], 255];
            }
        }
        Variant::Cyclic => {
            let k = sim.cyclic_states.max(2) as usize;
            pal[0] = [10, 10, 15, 255];
            for s in 0..k.min(256) {
                let hue = s as f32 / k as f32;
                let rgb = hsv_to_rgb(hue, 0.95, 1.0);
                pal[s] = [rgb[0], rgb[1], rgb[2], 255];
            }
        }
        Variant::GrayScott => {
            pal[0] = [10, 10, 15, 255];
            pal[1] = [0, 200, 120, 255];
            pal[2] = [0, 120, 255, 255];
            pal[3] = [255, 80, 60, 255];
            pal[4] = [255, 220, 0, 255];
        }
        _ => {}
    }

    pal
}

// -----------------------------
// Tiny bitmap font (5x7) + overlay draw into upload buffer
// -----------------------------
fn glyph_5x7(c: char) -> [u8; 7] {
    match c {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'J' => [0b11111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],

        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],

        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        '%' => [0b11001, 0b11010, 0b00100, 0b01000, 0b10110, 0b00110, 0b00000],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
        '\'' => [0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        _ => [0, 0, 0, 0, 0, 0, 0],
    }
}

fn draw_text_5x7_rgba(
    img: &mut [u8],
    bpr: usize,
    w: i32,
    h: i32,
    mut x: i32,
    mut y: i32,
    text: &str,
    scale: i32,
    rgba: [u8; 4],
) {
    let scale = scale.max(1);
    for ch in text.chars() {
        let c = if ch.is_ascii_lowercase() {
            (ch as u8 as char).to_ascii_uppercase()
        } else {
            ch
        };
        let g = glyph_5x7(c);

        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if (bits >> (4 - col)) & 1 == 0 {
                    continue;
                }
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = x + col as i32 * scale + sx;
                        let py = y + row as i32 * scale + sy;
                        if px < 0 || py < 0 || px >= w || py >= h {
                            continue;
                        }
                        let off = py as usize * bpr + px as usize * 4;
                        if off + 3 < img.len() {
                            img[off] = rgba[0];
                            img[off + 1] = rgba[1];
                            img[off + 2] = rgba[2];
                            img[off + 3] = rgba[3];
                        }
                    }
                }
            }
        }

        // spacing: 1 px (scaled)
        x += (5 + 1) * scale;
        if x > w {
            x = 0;
            y += (7 + 2) * scale;
        }
    }
}

fn text_px_width_5x7(text: &str, scale: i32) -> i32 {
    let scale = scale.max(1);
    let n = text.chars().count() as i32;
    if n <= 0 {
        0
    } else {
        n * (5 + 1) * scale
    }
}

fn text_px_height_5x7(lines: usize, scale: i32) -> i32 {
    let scale = scale.max(1);
    (lines as i32) * (7 + 2) * scale
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

struct Overlay<'a> {
    // debug lines (top-left on a visible monitor)
    lines: &'a [String],
    // flash big centered on visible rect
    flash: Option<&'a str>,
    // anchor visible rect (x,y,w,h) in window coords
    visible_rect: (i32, i32, i32, i32),
}

impl Gfx {
    async fn new(window: Arc<Window>, width: u32, height: u32) -> Self {
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

        let limits = wgpu::Limits::default();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
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
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (tex, tex_view, tex_w, tex_h, bpr, upload) =
            Self::make_pixel_texture(&device, config.width, config.height);

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

    fn make_pixel_texture(
        device: &wgpu::Device,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, u32, u32, u32, Vec<u8>) {
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

    fn resize(&mut self, new_w: u32, new_h: u32) {
        self.config.width = new_w.max(1);
        self.config.height = new_h.max(1);
        self.surface.configure(&self.device, &self.config);

        let (tex, tex_view, tex_w, tex_h, bpr, upload) =
            Self::make_pixel_texture(&self.device, self.config.width, self.config.height);

        self.tex = tex;
        self.tex_view = tex_view;
        self.tex_w = tex_w;
        self.tex_h = tex_h;
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

    fn upload_pixels(&mut self, sim: &Sim, palette: &[[u8; 4]; 256], overlay: Option<Overlay<'_>>) {
        let w = sim.w.min(self.tex_w as usize);
        let h = sim.h.min(self.tex_h as usize);

        let bpr = self.bpr as usize;
        let cur = &sim.cur;
        let upload = &mut self.upload;

        upload.par_chunks_mut(bpr).enumerate().take(h).for_each(|(y, row)| {
            let base = y * sim.w;
            let mut off = 0usize;
            for x in 0..w {
                let v = cur[base + x] as usize;
                let c = palette[v.min(255)];
                row[off] = c[0];
                row[off + 1] = c[1];
                row[off + 2] = c[2];
                row[off + 3] = c[3];
                off += 4;
            }
        });

        // Draw overlay (single-threaded, after sim blit)
        if let Some(ov) = overlay {
            let img_w = self.tex_w as i32;
            let img_h = self.tex_h as i32;
            let bpr_usz = self.bpr as usize;

            // Debug info top-left on visible monitor rect
            let (rx, ry, _rw, _rh) = ov.visible_rect;
            let pad = 12;
            let mut x0 = rx + pad;
            let mut y0 = ry + pad;

            // Clamp into image bounds a bit
            x0 = x0.clamp(0, img_w - 1);
            y0 = y0.clamp(0, img_h - 1);

            let scale_small = 2;
            let color = [255, 220, 0, 255];

            for line in ov.lines {
                draw_text_5x7_rgba(
                    &mut self.upload,
                    bpr_usz,
                    img_w,
                    img_h,
                    x0,
                    y0,
                    line,
                    scale_small,
                    color,
                );
                y0 += (7 + 2) * scale_small;
            }

            // Flash centered on visible rect (primary monitor)
            if let Some(txt) = ov.flash {
                let scale_big = 4;
                let tw = text_px_width_5x7(txt, scale_big);
                let th = text_px_height_5x7(1, scale_big);
                let (rx, ry, rw, rh) = ov.visible_rect;

                let cx = rx + (rw - tw) / 2;
                let cy = ry + (rh - th) / 2;

                // White for flash
                draw_text_5x7_rgba(
                    &mut self.upload,
                    bpr_usz,
                    img_w,
                    img_h,
                    cx,
                    cy,
                    txt,
                    scale_big,
                    [255, 255, 255, 255],
                );
            }
        }

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
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });

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

    user_brush_size: f32,
    user_paint_density: f32, 
    paint_accum: f32,
    paint_hz: f32,

    // debug overlay positioning: visible rect on a real monitor
    visible_rect: (i32, i32, i32, i32),

    // variant switching state
    last_variant: Variant,
    variant_idx: usize,

    // flash overlay
    flash_text: Option<String>,
    flash_until: Option<Instant>,

    // debug stats cache (for overlay)
    dbg_cpu: f32,
    dbg_mem_pct: f32,
    dbg_fps: u64,
    dbg_sps: u64,

    // overlay lines buffer
    overlay_lines: Vec<String>,

    // time
    t0: Instant,
    last_frame: Instant,
    sim_accum: f32,

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
            user_brush_size: 18.0,
            user_paint_density: 0.03,
            paint_accum: 0.0,
            paint_hz: 60.0,

            visible_rect: (0, 0, 1280, 720),

            last_variant: Variant::Standard,
            variant_idx: 0,

            flash_text: None,
            flash_until: None,

            dbg_cpu: 0.0,
            dbg_mem_pct: 0.0,
            dbg_fps: 0,
            dbg_sps: 0,

            overlay_lines: Vec::with_capacity(16),

            t0: Instant::now(),
            last_frame: Instant::now(),
            sim_accum: 0.0,

            last_stat: Instant::now(),
            frames: 0,
            sim_steps: 0,
            sys,
        }
    }

    fn set_flash_variant(&mut self, v: Variant) {
        if self.debug {
            self.flash_text = Some(v.label().to_string());
            self.flash_until = Some(Instant::now() + Duration::from_secs(2));
        }
    }

    fn force_variant(&mut self, now: f32, v: Variant) {
        // We compute everything while sim/ap are mutably borrowed,
        // but we call self.set_flash_variant(..) only AFTER the borrows end.
        let flash_v: Variant = {
            let (sim, ap) = match (&mut self.sim, &mut self.autopilot) {
                (Some(s), Some(a)) => (s, a),
                _ => return,
            };

            sim.variant = v;

            match v {
                Variant::Cyclic => sim.cyclic_states = ap.rng.random_range(8..=18),
                Variant::Generations => sim.gen_decay_max = ap.rng.random_range(6..=14),
                Variant::BriansBrain => sim.bb_refractory_max = ap.rng.random_range(5..=9),
                _ => {}
            }

            for k in 1..=4 {
                let col = ap.rand_color();
                let dur = ap.rng.random_range(4.0..10.0);
                ap.pal0[k].set_target(now, col, dur);
            }

            sim.clear();
            sim.seed_random(&mut ap.rng);

            ap.reset_at = now + 60.0;

            self.last_variant = sim.variant;
            self.variant_idx = Variant::all()
                .iter()
                .position(|&vv| vv == sim.variant)
                .unwrap_or(0);

            sim.variant
        };

        // sim/ap borrows are dropped here ✅
        self.set_flash_variant(flash_v);
    }


    fn build_overlay_lines(&self, sim: &Sim, ap: &AutoPilot, speed_ips: f32) -> Vec<String> {
        let mut lines = Vec::new();

        lines.push(format!(
            "Variant: {:?}   Speed: {:.2} ips",
            sim.variant, speed_ips
        ));
        lines.push(format!(
            "User brush: {:.0}   User density: {:.3}",
            self.user_brush_size, self.user_paint_density
        ));
        lines.push(format!(
            "Auto density: {:.3}   Auto rate: {:.3}",
            ap.step_cursor_density.cur, ap.step_cursor_rate.cur // Fixed names here
        ));
        lines.push(format!(
            "CPU {:4.1}%  MEM {:4.1}%",
            self.dbg_cpu, self.dbg_mem_pct
        ));
        lines.push(format!(
            "FPS {}  steps/s {}",
            self.dbg_fps, self.dbg_sps
        ));

        lines
    }

}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        #[cfg(windows)]
        let (vx, vy, vw, vh, rects) = {
            let rects = monitor_rects();
            if rects.is_empty() {
                (0i32, 0i32, 1280i32, 720i32, rects)
            } else {
                let (x, y, w, h) = bbox(&rects);
                (x, y, w, h, rects)
            }
        };

        #[cfg(not(windows))]
        let (vx, vy, vw, vh) = (0i32, 0i32, 1280i32, 720i32);

        // Determine a "visible" rect where we place debug (primary monitor if possible)
        #[cfg(windows)]
        {
            let pick = rects
                .iter()
                .find(|m| m.primary)
                .copied()
                .or_else(|| rects.first().copied());

            if let Some(m) = pick {
                // convert to window coords
                let rx = m.l - vx;
                let ry = m.t - vy;
                let rw = (m.r - m.l).max(1);
                let rh = (m.b - m.t).max(1);
                self.visible_rect = (rx, ry, rw, rh);
            } else {
                self.visible_rect = (0, 0, vw.max(1), vh.max(1));
            }
        }

        #[cfg(not(windows))]
        {
            self.visible_rect = (0, 0, vw.max(1), vh.max(1));
        }

        let mut attrs = WindowAttributes::default()
            .with_title("Conway Screensaver")
            .with_decorations(!self.is_screensaver)
            .with_resizable(!self.is_screensaver);

        if self.is_screensaver {
            attrs = attrs
                .with_inner_size(PhysicalSize::new(vw as u32, vh as u32))
                .with_position(PhysicalPosition::new(vx, vy));
        } else {
            attrs = attrs.with_inner_size(PhysicalSize::new(1280u32, 720u32));
            self.visible_rect = (0, 0, 1280, 720);
        }

        let win = Arc::new(el.create_window(attrs).expect("create_window"));
        win.set_cursor_visible(true);
        let _ = win.set_cursor_grab(CursorGrabMode::None);

        let size = win.inner_size();
        let gfx = block_on(Gfx::new(win.clone(), size.width, size.height));

        let sw = size.width as usize;
        let sh = size.height as usize;

        let mut sim = Sim::new(sw, sh);
        let mut ap = AutoPilot::new(sim.w, sim.h, 3);

        ap.reset(0.0, &mut sim);

        self.variant_idx = Variant::all()
            .iter()
            .position(|&v| v == sim.variant)
            .unwrap_or(0);
        self.last_variant = sim.variant;

        self.window = Some(win);
        self.gfx = Some(gfx);
        self.sim = Some(sim);
        self.autopilot = Some(ap);

        self.t0 = Instant::now();
        self.last_frame = Instant::now();
        self.last_stat = Instant::now();
        self.sim_accum = 0.0;
        self.frames = 0;
        self.sim_steps = 0;

        self.dbg_cpu = 0.0;
        self.dbg_mem_pct = 0.0;
        self.dbg_fps = 0;
        self.dbg_sps = 0;

        // Flash variant at start if debug is on
        if let Some(sim) = &self.sim {
            self.set_flash_variant(sim.variant);
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => el.exit(),

                        // Debug toggle
                        Key::Character(ref s) if s.as_str().eq_ignore_ascii_case("d") => {
                            self.debug = !self.debug;
                            if self.debug {
                                if let Some(sim) = &self.sim {
                                    self.set_flash_variant(sim.variant);
                                }
                            } else {
                                self.flash_text = None;
                                self.flash_until = None;
                            }
                        }

                        // Variant switching
                        Key::Named(NamedKey::ArrowRight) => {
                            let all = Variant::all();
                            self.variant_idx = (self.variant_idx + 1) % all.len();
                            let now = self.t0.elapsed().as_secs_f32();
                            self.force_variant(now, all[self.variant_idx]);
                        }
                        Key::Named(NamedKey::ArrowLeft) => {
                            let all = Variant::all();
                            self.variant_idx = if self.variant_idx == 0 {
                                all.len() - 1
                            } else {
                                self.variant_idx - 1
                            };
                            let now = self.t0.elapsed().as_secs_f32();
                            self.force_variant(now, all[self.variant_idx]);
                        }
                        // Reseed current variant
                        Key::Named(NamedKey::ArrowUp) => {
                            let now = self.t0.elapsed().as_secs_f32();
                            let v = self.last_variant;
                            self.force_variant(now, v);
                        }

                        // User density +/-
                        Key::Character(ref s) if s.as_str() == "+" || s.as_str() == "=" => {
                            self.user_paint_density =
                                (self.user_paint_density + 0.005).clamp(0.001, 0.25);
                        }
                        Key::Character(ref s) if s.as_str() == "-" || s.as_str() == "_" => {
                            self.user_paint_density =
                                (self.user_paint_density - 0.005).clamp(0.001, 0.25);
                        }

                        _ => {}
                    }
                }
            }

            WindowEvent::Resized(sz) => {
                if let Some(gfx) = &mut self.gfx {
                    gfx.resize(sz.width, sz.height);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor = Some((position.x as f32, position.y as f32));
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

        // ----------------------------
        // Frame timing
        // ----------------------------
        let now_i = Instant::now();
        let mut dt = (now_i - self.last_frame).as_secs_f32();
        self.last_frame = now_i;

        if dt.is_nan() || dt < 0.0 {
            dt = 0.0;
        }
        if dt > 0.25 {
            dt = 0.25;
        }

        let t = (now_i - self.t0).as_secs_f32();

        // ----------------------------
        // 1) Update autopilot params (once per frame)
        // ----------------------------
        let (speed_ips, base_pal, variant_now) = {
            let (sim, ap) = match (self.sim.as_mut(), self.autopilot.as_mut()) {
                (Some(s), Some(a)) => (s, a),
                _ => return,
            };

            // keep as you had it:
            ap.tick_cursor_motion(t, sim);

            let (speed_ips, base_pal) = ap.tick_frame(t, sim);
            (speed_ips, base_pal, sim.variant)
        };

        // ----------------------------
        // 2) 60Hz autopaint tick (python-like, decoupled from sim steps)
        // ----------------------------
        self.paint_accum += dt;
        let paint_step = 1.0 / self.paint_hz.max(1.0);

        while self.paint_accum >= paint_step {
            self.paint_accum -= paint_step;

            let (sim, ap) = match (self.sim.as_mut(), self.autopilot.as_mut()) {
                (Some(s), Some(a)) => (s, a),
                _ => break,
            };

            // NEW: this should do "one draw call per cursor per tick"
            // like python. (Implement this in AutoPilot.)
            ap.tick_paint_60hz(t, sim);
        }

        // ----------------------------
        // 3) Wheel -> brush size (as before)
        // ----------------------------
        if self.input.wheel_delta.abs() > 0.001 {
            self.user_brush_size = (self.user_brush_size + self.input.wheel_delta * 2.0).clamp(1.0, 150.0);
            self.input.wheel_delta = 0.0;
        }

        // ----------------------------
        // 4) Simulation stepping (dt-accumulated)
        // ----------------------------
        self.sim_accum += dt;
        let step_dt = 1.0 / speed_ips.max(1e-6);
        let dt_gs = step_dt.min(0.02);

        let mut steps_this_frame = 0usize;
        const MAX_STEPS_PER_FRAME: usize = 8;

        while self.sim_accum >= step_dt && steps_this_frame < MAX_STEPS_PER_FRAME {
            {
                let (sim, ap) = match (self.sim.as_mut(), self.autopilot.as_mut()) {
                    (Some(s), Some(a)) => (s, a),
                    _ => break,
                };

                // keep your existing behavior:
                ap.tick_step(speed_ips, sim);
                sim.step(dt_gs);

                // variant_now could change on reset:
                // (not strictly necessary, but keeps debug correct)
                // NOTE: we can't update `variant_now` here without re-borrows;
                // we’ll read it later when needed.
            }

            self.sim_accum -= step_dt;
            steps_this_frame += 1;
            self.sim_steps += 1;
        }

        // ----------------------------
        // 5) User painting (one draw per frame)
        // ----------------------------
        let do_user_paint = self.input.left_down || self.input.right_down;
        let cursor = self.input.cursor;
        let brush = self.user_brush_size.round() as i32;
        let user_fill = self.user_paint_density; // fill fraction semantics
        let left_down = self.input.left_down;
        let right_down = self.input.right_down;

        if do_user_paint {
            if let Some((mx, my)) = cursor {
                let gx = mx.round() as i32;
                let gy = my.round() as i32;

                let (sim, ap) = match (self.sim.as_mut(), self.autopilot.as_mut()) {
                    (Some(s), Some(a)) => (s, a),
                    _ => return,
                };

                // simplest semantics (still your old behavior):
                // left paints "alive", right erases.
                let val = if left_down { 1 } else { 0 };

                if sim.variant == Variant::GrayScott {
                    if val != 0 {
                        sim.seed_blob_gray_scott(gx, gy, brush.max(8));
                    }
                } else {
                    // your existing paint_disk uses "per-cell probability";
                    // python uses fill-fraction->k points; that's handled in autopaint_60hz.
                    // For user input, this is ok for now; if you want python-exact user paint
                    // switch to the same k-point scatter method later.
                    let density = user_fill.clamp(0.0, 1.0);
                    let mut rng = &mut ap.rng;
                    sim.paint_disk(&mut rng, gx, gy, brush, val, density);
                }

                // if both pressed, you can decide priority; current logic uses left if down else erase
                let _ = right_down; // keep variable used if you want custom logic
            }
        }

        // ----------------------------
        // 6) Render
        // ----------------------------
        // Expire flash text
        if let Some(until) = self.flash_until {
            if Instant::now() >= until {
                self.flash_text = None;
                self.flash_until = None;
            }
        }

        // Build overlay (only in debug)
        let overlay_opt = if self.debug {
            if let (Some(sim_ref), Some(ap_ref)) = (self.sim.as_ref(), self.autopilot.as_ref()) {
                self.overlay_lines = self.build_overlay_lines(sim_ref, ap_ref, speed_ips);
            } else {
                self.overlay_lines.clear();
            }

            Some(Overlay {
                lines: &self.overlay_lines,
                flash: self.flash_text.as_deref(),
                visible_rect: self.visible_rect,
            })
        } else {
            None
        };

        if let (Some(gfx), Some(sim_ref)) = (self.gfx.as_mut(), self.sim.as_ref()) {
            let pal = build_palette(sim_ref, &base_pal);
            gfx.upload_pixels(sim_ref, &pal, overlay_opt);
            gfx.render();
        }
        self.frames += 1;

        // ----------------------------
        // 7) Debug stats (optional)
        // ----------------------------
        if self.debug && self.last_stat.elapsed() >= Duration::from_secs(1) {
            self.sys.refresh_cpu_all();
            self.sys.refresh_memory();

            let cpu = self.sys.global_cpu_usage();
            let mem_used = self.sys.used_memory();
            let mem_total = self.sys.total_memory();
            let mem_pct = if mem_total > 0 {
                (mem_used as f32 / mem_total as f32) * 100.0
            } else {
                0.0
            };

            let fps = self.frames;
            let sps = self.sim_steps;

            self.dbg_cpu = cpu;
            self.dbg_mem_pct = mem_pct;
            self.dbg_fps = fps;
            self.dbg_sps = sps;

            // re-read variant and speed safely (no long borrows):
            let (variant, speed_cur) = match (self.sim.as_ref(), self.autopilot.as_ref()) {
                (Some(s), Some(a)) => (s.variant, a.speed_ips.cur),
                _ => (variant_now, speed_ips),
            };

            eprintln!(
                "[DEBUG] CPU {:5.1}% | MEM {:5.1}% | FPS {} | steps/s {} | variant {:?} | speed_ips {:.2} | user_brush {:.0} | user_fill {:.4}",
                cpu, mem_pct, fps, sps, variant, speed_cur, self.user_brush_size, self.user_paint_density
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