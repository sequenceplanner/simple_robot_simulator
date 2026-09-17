use crate::{SimulatorState, lock_simulator_state};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep};

/// How often the simulated arm takes a step, in milliseconds.
pub static SIM_RATE_MS: u64 = 10;

/// Positions are compared for exact equality to decide when the move is over, so
/// they are quantised first. Without this the loop would chase a target it can
/// never land on exactly.
const QUANTUM: f64 = 100000.0;

/// Below this share of the leading axis's error a joint stops being stepped and
/// snaps straight to its target, which is what keeps the trailing joints from
/// crawling for the rest of the move.
const SNAP_THRESHOLD: f64 = 0.1;

#[derive(Debug, PartialEq, Eq)]
pub enum MotionOutcome {
    Completed,
    Cancelled,
}

fn quantize(values: &[f64]) -> Vec<f64> {
    values.iter().map(|x| (*x * QUANTUM).round() / QUANTUM).collect()
}

/// Drive the actual joint positions towards `reference`, one step per tick.
///
/// All joints start and stop together: each joint's error is scaled by the largest
/// error in the set, so the leading axis moves a full step and the rest move a
/// proportional fraction of one. That is the same profile the ROS version had.
///
/// `acceleration` is deliberately not used. It never was - the old action accepted
/// it and ignored it - and inventing a ramp here would change how every existing
/// motion looks without anyone asking for it.
pub async fn simulate_movement(
    simulator_state: &Arc<Mutex<SimulatorState>>,
    reference: &[f64],
    velocity: f64,
    cancel: &mut mpsc::Receiver<()>,
    log_target: &str,
) -> Result<MotionOutcome, String> {
    let actual = lock_simulator_state(simulator_state).actual_joint_positions.clone();

    // The reference now comes from whatever wrote it to Redis, so it can disagree
    // with the robot the URDF describes. The step loop runs until the two vectors
    // are equal, so a length mismatch would never terminate.
    if reference.len() != actual.len() {
        return Err(format!(
            "expected {} joint positions for this robot, got {}",
            actual.len(),
            reference.len()
        ));
    }

    if !reference.iter().all(|x| x.is_finite()) {
        return Err("joint positions must all be finite".to_string());
    }

    // A velocity of zero would mean a zero step and a move that never ends.
    let velocity = if velocity == 0.0 { 1.0 } else { velocity.abs() };
    let step = 0.1 * velocity * std::f64::consts::PI / 180.0;

    let reference = quantize(reference);
    let mut actual = quantize(&actual);

    while actual != reference {
        if cancel.try_recv().is_ok() {
            log::info!(target: log_target, "Motion cancelled, holding position.");
            return Ok(MotionOutcome::Cancelled);
        }

        let errors: Vec<f64> = reference
            .iter()
            .zip(actual.iter())
            .map(|(r, a)| (r - a).abs())
            .collect();

        let max_error = errors.iter().copied().fold(f64::NAN, f64::max);
        let sync_factor = if max_error == 0.0 { 1.0 } else { 1.0 / max_error };

        let mut next = Vec::with_capacity(actual.len());
        for ((r, a), e) in reference
            .iter()
            .zip(actual.iter())
            .zip(errors.iter().map(|e| e * sync_factor))
        {
            let value = if e <= SNAP_THRESHOLD {
                *r
            } else if r < &(a - step) {
                a - step * e
            } else if r > &(a + step) {
                a + step * e
            } else {
                *r
            };
            next.push((value * QUANTUM).round() / QUANTUM);
        }

        actual = next;
        lock_simulator_state(simulator_state).actual_joint_positions = actual.clone();

        sleep(Duration::from_millis(SIM_RATE_MS)).await;
    }

    Ok(MotionOutcome::Completed)
}
