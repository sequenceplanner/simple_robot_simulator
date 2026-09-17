//! The interpolator: it has to converge, refuse a bad reference, and stop on cancel.

use simple_robot_simulator::*;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

fn state(positions: Vec<f64>) -> Arc<Mutex<SimulatorState>> {
    Arc::new(Mutex::new(SimulatorState::new(positions)))
}

#[tokio::test]
async fn converges_on_the_reference() {
    let sim = state(vec![0.0; 6]);
    let (_tx, mut rx) = mpsc::channel(1);
    let reference = vec![0.1, -0.1, 0.05, 0.0, 0.0, 0.0];

    let outcome = simulate_movement(&sim, &reference, 50.0, &mut rx, "test")
        .await
        .expect("a well-formed move should run");

    assert_eq!(outcome, MotionOutcome::Completed);
    assert_eq!(
        lock_simulator_state(&sim).actual_joint_positions,
        reference,
        "the arm has to land exactly on the reference, not near it"
    );
}

#[tokio::test]
async fn a_move_to_the_current_position_completes_immediately() {
    let positions = vec![0.5; 6];
    let sim = state(positions.clone());
    let (_tx, mut rx) = mpsc::channel(1);

    let outcome = simulate_movement(&sim, &positions, 1.0, &mut rx, "test")
        .await
        .unwrap();

    assert_eq!(outcome, MotionOutcome::Completed);
}

#[tokio::test]
async fn rejects_a_reference_of_the_wrong_length() {
    let sim = state(vec![0.0; 6]);
    let (_tx, mut rx) = mpsc::channel(1);

    // The old ROS code looped `while act != ref` over mismatched vectors, which never
    // terminates. The reference now comes from Redis, so this is reachable input.
    let result = simulate_movement(&sim, &[0.0, 0.0, 0.0], 1.0, &mut rx, "test").await;

    let error = result.expect_err("a three-element reference for a six-joint arm must fail");
    assert!(error.contains('6') && error.contains('3'), "got: {}", error);
}

#[tokio::test]
async fn rejects_a_non_finite_reference() {
    let sim = state(vec![0.0; 6]);
    let (_tx, mut rx) = mpsc::channel(1);

    let reference = vec![0.0, f64::NAN, 0.0, 0.0, 0.0, 0.0];
    let result = simulate_movement(&sim, &reference, 1.0, &mut rx, "test").await;

    assert!(
        result.is_err(),
        "NaN would make the equality check never succeed"
    );
}

#[tokio::test]
async fn a_zero_velocity_still_terminates() {
    let sim = state(vec![0.0; 6]);
    let (_tx, mut rx) = mpsc::channel(1);
    let reference = vec![0.02, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Zero would mean a zero step and a move that never ends, so it falls back to 1.0.
    let outcome = simulate_movement(&sim, &reference, 0.0, &mut rx, "test")
        .await
        .unwrap();

    assert_eq!(outcome, MotionOutcome::Completed);
}

#[tokio::test]
async fn cancel_stops_the_move_short() {
    let sim = state(vec![0.0; 6]);
    let (tx, mut rx) = mpsc::channel(1);
    // Far away and slow, so the move is still running when the cancel lands.
    let reference = vec![3.0; 6];

    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let _ = tx.send(()).await;
    });

    let outcome = simulate_movement(&sim, &reference, 0.1, &mut rx, "test")
        .await
        .unwrap();

    assert_eq!(outcome, MotionOutcome::Cancelled);
    assert_ne!(
        lock_simulator_state(&sim).actual_joint_positions,
        reference,
        "a cancelled move must stop where it is, not finish"
    );
}
