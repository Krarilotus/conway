// src/tuning.rs
//
// This file is the CONTROL PANEL.

use rand::Rng;

use crate::{ColorTransition, Sim, Transition, Variant};

#[derive(Clone, Copy, Debug)]
pub struct SecondsRange {
    /// Minimum seconds for a retarget/ease.
    pub min: f32,
    /// Maximum seconds for a retarget/ease.
    pub max: f32,
}
impl SecondsRange {
    #[inline]
    pub fn pick<R: Rng + ?Sized>(self, rng: &mut R) -> f32 {
        rng.random_range(self.min..self.max)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RangeF32 {
    pub min: f32,
    pub max: f32,
}
impl RangeF32 {
    #[inline]
    pub fn pick<R: Rng + ?Sized>(self, rng: &mut R) -> f32 {
        rng.random_range(self.min..self.max)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RangeI32 {
    pub min: i32,
    pub max: i32,
}
impl RangeI32 {
    #[inline]
    pub fn pick<R: Rng + ?Sized>(self, rng: &mut R) -> i32 {
        rng.random_range(self.min..self.max)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RangeU8 {
    pub min: u8,
    pub max: u8,
}
impl RangeU8 {
    #[inline]
    pub fn pick<R: Rng + ?Sized>(self, rng: &mut R) -> u8 {
        rng.random_range(self.min..=self.max)
    }
}

/// An animated “dial” for f32 values.
#[derive(Clone, Copy, Debug)]
pub struct AnimatedF32 {
    pub enabled: bool,
    pub initial: f32,
    pub range: RangeF32,
    pub retarget: SecondsRange,
}
impl AnimatedF32 {
    #[inline]
    pub fn make_transition(self) -> Transition {
        Transition::new(self.initial, self.range.min, self.range.max, self.retarget.min, self.retarget.max)
    }

    #[inline]
    pub fn maybe_retarget<R: Rng + ?Sized>(self, tr: &mut Transition, now: f32, rng: &mut R) {
        if !self.enabled {
            return;
        }
        tr.maybe_new_target(now, rng, |r| self.range.pick(r));
    }
}

/// Animated palette color.
#[derive(Clone, Copy, Debug)]
pub struct AnimatedColor {
    pub enabled: bool,
    pub initial: [u8; 3],
    pub component_min: u8,
    pub component_max: u8,
    pub retarget: SecondsRange,
}
impl AnimatedColor {
    #[inline]
    pub fn make_transition(self) -> ColorTransition {
        ColorTransition::new(self.initial, self.retarget.min, self.retarget.max)
    }

    #[inline]
    pub fn pick_random_rgb<R: Rng + ?Sized>(self, rng: &mut R) -> [u8; 3] {
        [
            rng.random_range(self.component_min..=self.component_max),
            rng.random_range(self.component_min..=self.component_max),
            rng.random_range(self.component_min..=self.component_max),
        ]
    }

    #[inline]
    pub fn maybe_retarget<R: Rng + ?Sized>(self, ct: &mut ColorTransition, now: f32, rng: &mut R) {
        if !self.enabled {
            return;
        }
        ct.maybe_new_target(now, rng, |r| self.pick_random_rgb(r));
    }
}


/// What happens when the autopilot “reset” fires.
#[derive(Clone, Copy, Debug)]
pub struct ResetPolicy {
    pub change_variant: bool,
    pub reseed_sim_state: bool,
    pub randomize_variant_params: bool,
    pub randomize_palette_on_reset: bool,
    pub recenter_cursors_on_reset: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PaintValuePolicy {
    pub immigration_species2_prob: f32,
    pub quadlife_weights: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct CursorMotionTuning {
    pub xy_retarget: SecondsRange,
    // Removed unused brush_retarget field
}

#[derive(Clone, Copy, Debug)]
pub struct CursorStampTuning {
    pub brush_radius_px: AnimatedF32,
    pub fill_fraction: AnimatedF32,
    pub layer_multiplier: AnimatedF32,
}

#[derive(Clone, Copy, Debug)]
pub struct CursorSystemTuning {
    pub motion: CursorMotionTuning,
    pub stamp_60hz: CursorStampTuning,
}

#[derive(Clone, Copy, Debug)]
pub struct BlobTuning {
    pub blobs_per_second: AnimatedF32,
    pub blob_radius_px: AnimatedF32,
    pub blob_cell_probability: AnimatedF32,
}

#[derive(Clone, Copy, Debug)]
pub struct StepCursorTuning {
    pub stamps_per_second: AnimatedF32,
    pub brush_radius_px: AnimatedF32,
    pub stamp_cell_probability: AnimatedF32,
}

#[derive(Clone, Copy, Debug)]
pub struct LargerThanLifeTuning {
    pub radius_r: AnimatedF32,
    pub birth_lo: AnimatedF32,
    pub birth_hi: AnimatedF32,
    pub survive_lo: AnimatedF32,
    pub survive_hi: AnimatedF32,
}

#[derive(Clone, Copy, Debug)]
pub struct PaletteTuning {
    pub background: AnimatedColor,
    pub accent: AnimatedColor,
}

#[derive(Clone, Copy, Debug)]
pub struct VariantTuning {
    pub initial_alive_probability: f32,
    pub immigration_p0: f32,
    pub immigration_p1: f32,
    #[allow(dead_code)] // Implicitly used by logic (p2 = 1.0 - p0 - p1)
    pub immigration_p2: f32,
    pub quadlife_dead_probability: f32,
    pub brians_brain_dead_probability: f32,
    pub cyclic_dead_probability: f32,
    pub cyclic_states: RangeU8,
    pub generations_decay_max: RangeU8,
    pub brians_brain_refractory_max: RangeU8,
    pub gray_scott_blob_count: usize,
    pub gray_scott_blob_radius_px: RangeI32,
}

impl Default for VariantTuning {
    fn default() -> Self {
        Self {
            initial_alive_probability: 0.14,
            immigration_p0: 0.82,
            immigration_p1: 0.09,
            immigration_p2: 0.09,
            quadlife_dead_probability: 0.86,
            brians_brain_dead_probability: 0.92,
            cyclic_dead_probability: 0.90,
            cyclic_states: RangeU8 { min: 8, max: 18 },
            generations_decay_max: RangeU8 { min: 6, max: 14 },
            brians_brain_refractory_max: RangeU8 { min: 5, max: 9 },
            gray_scott_blob_count: 10,
            gray_scott_blob_radius_px: RangeI32 { min: 20, max: 80 },
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControlPanel {
    pub rng_seed: u64,
    pub reset_after_seconds: f32,
    pub reset_policy: ResetPolicy,
    pub sim_steps_per_second: AnimatedF32,
    pub blobs: BlobTuning,
    pub step_cursor: StepCursorTuning,
    pub cursors: CursorSystemTuning,
    pub ltl: LargerThanLifeTuning,
    pub palette: PaletteTuning,
    pub variants: VariantTuning,
    pub paint_values: PaintValuePolicy,
}

impl ControlPanel {
    pub fn for_sim(w: usize, h: usize) -> Self {
        let brush_max = ((w.min(h) as f32) / 5.0).max(10.0);
        let retarget_default = SecondsRange { min: 4.0, max: 10.0 };

        Self {
            rng_seed: 0xC0FFEE_1234_5678,

            reset_after_seconds: 60.0,
            reset_policy: ResetPolicy {
                change_variant: true,
                reseed_sim_state: true,
                randomize_variant_params: true,
                randomize_palette_on_reset: true,
                recenter_cursors_on_reset: false,
            },

            sim_steps_per_second: AnimatedF32 {
                enabled: true,
                initial: 12.0,
                range: RangeF32 { min: 5.0, max: 20.0 },
                retarget: retarget_default,
            },

            blobs: BlobTuning {
                blobs_per_second: AnimatedF32 {
                    enabled: true,
                    initial: 0.12,
                    range: RangeF32 { min: 0.04, max: 0.40 },
                    retarget: retarget_default,
                },
                blob_radius_px: AnimatedF32 {
                    enabled: true,
                    initial: 30.0,
                    range: RangeF32 { min: 10.0, max: 90.0 },
                    retarget: retarget_default,
                },
                blob_cell_probability: AnimatedF32 {
                    enabled: true,
                    initial: 0.030,
                    range: RangeF32 { min: 0.008, max: 0.070 },
                    retarget: retarget_default,
                },
            },

            step_cursor: StepCursorTuning {
                stamps_per_second: AnimatedF32 {
                    enabled: true,
                    initial: 0.7,
                    range: RangeF32 { min: 0.15, max: 1.6 },
                    retarget: retarget_default,
                },
                brush_radius_px: AnimatedF32 {
                    enabled: true,
                    initial: 20.0,
                    range: RangeF32 { min: 10.0, max: brush_max },
                    retarget: retarget_default,
                },
                stamp_cell_probability: AnimatedF32 {
                    enabled: true,
                    initial: 0.010,
                    range: RangeF32 { min: 0.0025, max: 0.030 },
                    retarget: retarget_default,
                },
            },

            cursors: CursorSystemTuning {
                motion: CursorMotionTuning {
                    xy_retarget: SecondsRange { min: 4.0, max: 10.0 },
                },
                stamp_60hz: CursorStampTuning {
                    brush_radius_px: AnimatedF32 {
                        enabled: true,
                        initial: 20.0,
                        range: RangeF32 { min: 10.0, max: brush_max },
                        retarget: SecondsRange { min: 4.0, max: 10.0 },
                    },
                    fill_fraction: AnimatedF32 {
                        enabled: true,
                        initial: 0.006,
                        range: RangeF32 { min: 0.001, max: 0.010 },
                        retarget: SecondsRange { min: 4.0, max: 10.0 },
                    },
                    layer_multiplier: AnimatedF32 {
                        enabled: true,
                        initial: 1.0,
                        range: RangeF32 { min: 0.5, max: 1.5 },
                        retarget: SecondsRange { min: 4.0, max: 10.0 },
                    },
                },
            },

            ltl: LargerThanLifeTuning {
                radius_r: AnimatedF32 {
                    enabled: true,
                    initial: 5.0,
                    range: RangeF32 { min: 2.0, max: 7.0 },
                    retarget: retarget_default,
                },
                birth_lo: AnimatedF32 {
                    enabled: true,
                    initial: 0.22,
                    range: RangeF32 { min: 0.16, max: 0.30 },
                    retarget: retarget_default,
                },
                birth_hi: AnimatedF32 {
                    enabled: true,
                    initial: 0.34,
                    range: RangeF32 { min: 0.22, max: 0.40 },
                    retarget: retarget_default,
                },
                survive_lo: AnimatedF32 {
                    enabled: true,
                    initial: 0.18,
                    range: RangeF32 { min: 0.12, max: 0.26 },
                    retarget: retarget_default,
                },
                survive_hi: AnimatedF32 {
                    enabled: true,
                    initial: 0.48,
                    range: RangeF32 { min: 0.22, max: 0.55 },
                    retarget: retarget_default,
                },
            },

            palette: PaletteTuning {
                background: AnimatedColor {
                    enabled: false,
                    initial: [10, 10, 15],
                    component_min: 5,
                    component_max: 25,
                    retarget: SecondsRange { min: 8.0, max: 14.0 },
                },
                accent: AnimatedColor {
                    enabled: true,
                    initial: [0, 255, 128],
                    component_min: 40,
                    component_max: 255,
                    retarget: SecondsRange { min: 4.0, max: 10.0 },
                },
            },

            variants: VariantTuning::default(),

            paint_values: PaintValuePolicy {
                immigration_species2_prob: 0.5,
                quadlife_weights: [1.0, 1.0, 1.0, 1.0],
            },
        }
    }
}

// --------------------------
// Variant helpers
// --------------------------

pub fn randomize_variant_params<R: Rng + ?Sized>(sim: &mut Sim, rng: &mut R, vt: VariantTuning) {
    match sim.variant {
        Variant::Cyclic => sim.cyclic_states = vt.cyclic_states.pick(rng),
        Variant::Generations => sim.gen_decay_max = vt.generations_decay_max.pick(rng),
        Variant::BriansBrain => sim.bb_refractory_max = vt.brians_brain_refractory_max.pick(rng),
        _ => {}
    }
}

pub fn seed_sim<R: Rng + ?Sized>(sim: &mut Sim, rng: &mut R, vt: VariantTuning) {
    match sim.variant {
        Variant::Immigration => {
            for v in &mut sim.cur {
                let r: f32 = rng.random();
                *v = if r < vt.immigration_p0 {
                    0
                } else if r < vt.immigration_p0 + vt.immigration_p1 {
                    1
                } else {
                    2
                };
            }
        }
        Variant::QuadLife => {
            for v in &mut sim.cur {
                let r: f32 = rng.random();
                *v = if r < vt.quadlife_dead_probability { 0 } else { rng.random_range(1..=4) };
            }
        }
        Variant::BriansBrain => {
            for v in &mut sim.cur {
                let r: f32 = rng.random();
                *v = if r < vt.brians_brain_dead_probability { 0 } else { 1 };
            }
        }
        Variant::Cyclic => {
            let k = sim.cyclic_states.max(2);
            for v in &mut sim.cur {
                let r: f32 = rng.random();
                *v = if r < vt.cyclic_dead_probability {
                    0
                } else {
                    rng.random_range(0..k) as u8
                };
            }
        }
        Variant::GrayScott => {
            sim.gs_u.fill(1.0);
            sim.gs_v.fill(0.0);
            for _ in 0..vt.gray_scott_blob_count {
                let cx = rng.random_range(0..sim.w) as i32;
                let cy = rng.random_range(0..sim.h) as i32;
                let rad = vt.gray_scott_blob_radius_px.pick(rng);
                sim.seed_blob_gray_scott(cx, cy, rad);
            }
            sim.update_gray_scott_cur_from_v();
        }
        _ => {
            for v in &mut sim.cur {
                let r: f32 = rng.random();
                *v = if r < vt.initial_alive_probability { 1 } else { 0 };
            }
        }
    }
}

pub fn pick_paint_value<R: Rng + ?Sized>(
    rng: &mut R,
    sim: &Sim,
    policy: PaintValuePolicy,
) -> u8 {
    match sim.variant {
        Variant::Immigration => {
            if rng.random::<f32>() < policy.immigration_species2_prob { 2 } else { 1 }
        }
        Variant::QuadLife => {
            let w = policy.quadlife_weights;
            let sum = (w[0] + w[1] + w[2] + w[3]).max(1e-9);
            let mut x = rng.random::<f32>() * sum;
            for (i, wi) in w.iter().enumerate() {
                x -= *wi;
                if x <= 0.0 {
                    return (i as u8) + 1;
                }
            }
            4
        }
        Variant::BriansBrain => 1,
        Variant::Cyclic => rng.random_range(0..sim.cyclic_states) as u8,
        _ => 1,
    }
}