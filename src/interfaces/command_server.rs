use crate::*;
use micro_sp::*;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Poll Redis for motion requests and serve them.
///
/// The handshake is `ur_redis_driver`'s, key for key: write the parameters, set
/// `{id}_request_trigger`, and watch `{id}_request_state` walk
/// `initial -> executing -> succeeded | failed`. A request always reaches a terminal
/// state - a rejected one fails rather than sitting at `initial` forever.
pub async fn command_server(
    robot_name: &str,
    tf_prefix: &str,
    chain: k::Chain<f64>,
    connection_manager: &Arc<ConnectionManager>,
    simulator_state: Arc<Mutex<SimulatorState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let log_target = format!("{}_command_server", robot_name);

    // `runner_interval` already sets `MissedTickBehavior::Delay`. The default is
    // `Burst`, which turns one slow Redis reply into a spin that starves every other
    // task on the runtime.
    let mut interval = runner_interval();

    let suffixes = [
        "request_trigger",
        "request_state",
        "request_result",
        "request_cancel",
        "request_feedback",
        "command_type",
        "acceleration",
        "velocity",
        "global_acceleration_scaling",
        "global_velocity_scaling",
        "use_execution_time",
        "execution_time",
        "use_blend_radius",
        "blend_radius",
        "use_joint_positions",
        "joint_positions",
        "use_preferred_joint_config",
        "preferred_joint_config",
        "use_payload",
        "payload",
        "baseframe_id",
        "faceplate_id",
        "goal_feature_id",
        "tcp_id",
        "root_frame_id",
        "use_relative_pose",
        "relative_pose",
        "force_threshold",
    ];

    let keys: Vec<String> = suffixes
        .iter()
        .map(|s| format!("{robot_name}_{s}"))
        .collect();

    // Nothing in the loop body happens unless one of these two is set, so an idle
    // tick reads two keys instead of the whole set above.
    let fast_keys: Vec<String> = ["request_cancel", "request_trigger"]
        .iter()
        .map(|s| format!("{robot_name}_{s}"))
        .collect();

    // One long-lived handle for the whole task. `SPConnection` is cheap to clone,
    // multiplexed and self-healing, so it stays valid across reconnects and there is
    // no reason to pre-flight a PING every tick.
    let mut con = connection_manager.get_connection().await;

    let key = |suffix: &str| format!("{robot_name}_{suffix}");

    loop {
        interval.tick().await;

        let Some(flags) = StateManager::get_state_for_keys(&mut con, &fast_keys, &log_target).await
        else {
            continue;
        };

        // Both of these are written by whoever drives the simulator, so a malformed
        // value is an input error, not an invariant violation. The plain accessors
        // panic on a key that failed to deserialize; these do not.
        let cancel_requested = state_bool_or(&flags, &key("request_cancel"), false, &log_target);
        let triggered = state_bool_or(&flags, &key("request_trigger"), false, &log_target);
        if !cancel_requested && !triggered {
            continue;
        }

        let Some(state) = StateManager::get_state_for_keys(&mut con, &keys, &log_target).await
        else {
            continue;
        };

        if state_bool_or(&state, &key("request_cancel"), false, &log_target) {
            StateManager::set_sp_value(&mut con, &key("request_cancel"), &false.to_spvalue()).await;

            let cancel_sender = lock_simulator_state(&simulator_state).cancel_sender.take();
            match cancel_sender {
                Some(sender) => {
                    log::info!(target: &log_target, "Cancel requested, stopping the active motion.");
                    let _ = sender.try_send(());
                }
                None => {
                    log::warn!(target: &log_target, "Cancel requested, but no goal is running.");
                }
            }
        }

        if !state_bool_or(&state, &key("request_trigger"), false, &log_target) {
            continue;
        }

        // Consume the edge before doing anything else, so a request cannot be served
        // twice. Every path from here on must write a terminal state.
        StateManager::set_sp_value(&mut con, &key("request_trigger"), &false.to_spvalue()).await;

        let request_state = state_string_or(
            &state,
            &key("request_state"),
            &ActionRequestState::UNKNOWN.to_string(),
            &log_target,
        );
        if request_state != ActionRequestState::Initial.to_string() {
            continue;
        }

        // `build_state` drops any key whose stored value will not deserialize, and
        // every accessor panics on a key that is absent. Checking the whole set once
        // turns a malformed parameter into a failed request instead of a dead
        // simulator, and reports every bad key at once rather than the first.
        let missing: Vec<&str> = keys
            .iter()
            .filter(|k| !state.contains(*k))
            .map(|k| k.as_str())
            .collect();
        if !missing.is_empty() {
            fail_request(
                &mut con,
                robot_name,
                &format!("missing or unreadable request keys: {}", missing.join(", ")),
                &log_target,
            )
            .await;
            continue;
        }

        let command_type = state_string_or(&state, &key("command_type"), "UNKNOWN", &log_target);

        if NOOP_COMMANDS.contains(&command_type.as_str()) {
            log::info!(
                target: &log_target,
                "'{}' has no effect on a kinematic simulator, succeeding immediately.",
                command_type
            );
            succeed_request(
                &mut con,
                robot_name,
                &format!("'{}' is a no-op in simulation", command_type),
                &log_target,
            )
            .await;
            continue;
        }

        if !MOTION_COMMANDS.contains(&command_type.as_str()) {
            fail_request(
                &mut con,
                robot_name,
                &format!("unknown command_type '{}'", command_type),
                &log_target,
            )
            .await;
            continue;
        }

        let velocity = state_float_or(&state, &key("velocity"), 0.0, &log_target);
        // `acceleration` is read as part of the key-presence check above but never
        // used: the motion profile is constant-velocity, as it was under ROS.
        let use_joint_positions =
            state_bool_or(&state, &key("use_joint_positions"), false, &log_target);

        // Prefixed, so a simulator whose interface state has not been seeded falls
        // back to *its own* base and faceplate. Falling back to the bare names would
        // have a second robot resolving goals against the first robot's arm.
        let baseframe_id = state_string_or(
            &state,
            &key("baseframe_id"),
            &format!("{}{}", tf_prefix, DEFAULT_BASEFRAME_ID),
            &log_target,
        );
        let faceplate_id = state_string_or(
            &state,
            &key("faceplate_id"),
            &format!("{}{}", tf_prefix, DEFAULT_FACEPLATE_ID),
            &log_target,
        );
        let goal_feature_id =
            state_string_or(&state, &key("goal_feature_id"), "UNKNOWN", &log_target);
        let tcp_id = state_string_or(&state, &key("tcp_id"), "UNKNOWN", &log_target);

        let joint_positions = extract_f64_array(&state, &key("joint_positions"), &log_target);

        // Admission control. A rejection must still write a terminal state: the
        // trigger is already consumed, so a caller waiting on `request_state` would
        // otherwise wait forever.
        if lock_simulator_state(&simulator_state).goal_id.is_some() {
            fail_request(
                &mut con,
                robot_name,
                "a goal is already running",
                &log_target,
            )
            .await;
            continue;
        }

        // Resolve the target joint configuration: either straight from the request,
        // or by looking the goal frame up in the transform buffer and solving IK.
        let reference = if use_joint_positions {
            match joint_positions {
                Some(positions) => positions,
                None => {
                    fail_request(
                        &mut con,
                        robot_name,
                        "use_joint_positions is set but joint_positions is not an array of floats",
                        &log_target,
                    )
                    .await;
                    continue;
                }
            }
        } else {
            // Where the goal frame sits in the robot's base frame.
            let target_in_base = match TransformsManager::lookup_transform(
                &mut con,
                &baseframe_id,
                &goal_feature_id,
            )
            .await
            {
                Ok(transform) => transform,
                Err(e) => {
                    fail_request(
                        &mut con,
                        robot_name,
                        &format!(
                            "no transform from '{}' to '{}': {}",
                            baseframe_id, goal_feature_id, e
                        ),
                        &log_target,
                    )
                    .await;
                    continue;
                }
            };

            // Where the tool centre point sits in the face plate frame. This is what
            // the arm is actually holding, so it is looked up per request rather than
            // baked into the URDF.
            let tcp_in_faceplate =
                match TransformsManager::lookup_transform(&mut con, &faceplate_id, &tcp_id).await {
                    Ok(transform) => transform,
                    Err(e) => {
                        fail_request(
                            &mut con,
                            robot_name,
                            &format!(
                                "no transform from '{}' to '{}': {}",
                                faceplate_id, tcp_id, e
                            ),
                            &log_target,
                        )
                        .await;
                        continue;
                    }
                };

            let Some(new_chain) =
                generate_new_kinematic_chain(&chain, &faceplate_id, &tcp_id, &tcp_in_faceplate)
            else {
                fail_request(
                    &mut con,
                    robot_name,
                    "failed to graft the tcp joint onto the kinematic chain",
                    &log_target,
                )
                .await;
                continue;
            };

            let seed = lock_simulator_state(&simulator_state)
                .actual_joint_positions
                .clone();

            publish_request_feedback(
                &mut con,
                robot_name,
                "computing inverse kinematics",
            )
            .await;

            match calculate_inverse_kinematics(
                &new_chain,
                &faceplate_id,
                &tcp_id,
                &target_in_base,
                &seed,
                &log_target,
            ) {
                Some(solution) => solution,
                None => {
                    fail_request(
                        &mut con,
                        robot_name,
                        &format!("no inverse kinematics solution for '{}'", goal_feature_id),
                        &log_target,
                    )
                    .await;
                    continue;
                }
            }
        };

        let goal_id = nanoid::nanoid!(10, &NANOID_ALPHABET);
        let (cancel_sender, mut cancel_receiver) = mpsc::channel(1);
        {
            let mut sim = lock_simulator_state(&simulator_state);
            sim.goal_id = Some(goal_id.clone());
            sim.cancel_sender = Some(cancel_sender);
        }

        StateManager::set_sp_value(
            &mut con,
            &key("request_state"),
            &ActionRequestState::Executing.to_string().to_spvalue(),
        )
        .await;
        publish_request_feedback(&mut con, robot_name, "moving").await;

        let simulator_state_clone = simulator_state.clone();
        let mut con_clone = con.clone();
        let robot_name = robot_name.to_string();
        let log_target_clone = log_target.clone();
        let linear = is_linear_command(&command_type);

        // The move blocks until it finishes, so it runs in its own task: the poll
        // loop has to stay responsive to a cancel while the arm is moving.
        tokio::spawn(async move {
            let outcome = simulate_movement(
                &simulator_state_clone,
                &reference,
                velocity,
                &mut cancel_receiver,
                &log_target_clone,
            )
            .await;

            match outcome {
                Ok(MotionOutcome::Completed) => {
                    let mut message = "motion completed".to_string();
                    // A joint-interpolated path reaches the same target by a
                    // different route. Say so rather than let a rehearsal imply the
                    // real arm will travel that way.
                    if linear {
                        message.push_str(
                            " (linear motion was approximated in joint space)",
                        );
                    }
                    succeed_request(&mut con_clone, &robot_name, &message, &log_target_clone).await;
                }
                Ok(MotionOutcome::Cancelled) => {
                    // A cancel that took effect is a successful cancel, matching
                    // `ur_redis_driver`.
                    succeed_request(
                        &mut con_clone,
                        &robot_name,
                        "motion cancelled",
                        &log_target_clone,
                    )
                    .await;
                }
                Err(reason) => {
                    fail_request(&mut con_clone, &robot_name, &reason, &log_target_clone).await;
                }
            }

            let mut sim = lock_simulator_state(&simulator_state_clone);
            sim.goal_id = None;
            sim.cancel_sender = None;
        });
    }
}

/// Read an `SPValue::Array` of floats.
///
/// `None` means the key held something that is not an array of numbers, which is a
/// request the caller got wrong rather than a value to substitute a default for.
fn extract_f64_array(state: &State, key: &str, log_target: &str) -> Option<Vec<f64>> {
    let Some(SPValue::Array(ArrayOrUnknown::Array(values))) = state.get_value(key, log_target)
    else {
        return None;
    };

    values
        .iter()
        .map(|value| match value {
            SPValue::Float64(FloatOrUnknown::Float64(f)) => Some(f.into_inner()),
            SPValue::Int64(IntOrUnknown::Int64(i)) => Some(*i as f64),
            _ => None,
        })
        .collect()
}

/// Finish a request that could not be served.
///
/// Every early exit in the loop above has already consumed `request_trigger`, so
/// without this the caller watches a `request_state` that will never leave
/// `initial`.
pub async fn fail_request(
    con: &mut SPConnection,
    robot_name: &str,
    reason: &str,
    log_target: &str,
) {
    log::error!(target: log_target, "Request failed: {}.", reason);

    let counter_keys = vec![
        format!("{robot_name}_total_fail_counter"),
        format!("{robot_name}_subsequent_fail_counter"),
    ];

    // Read-modify-write on the failure path only, so the happy path pays nothing.
    if let Some(counters) = StateManager::get_state_for_keys(con, &counter_keys, log_target).await {
        let total = state_int_or(&counters, &counter_keys[0], 0, log_target);
        let subsequent = state_int_or(&counters, &counter_keys[1], 0, log_target);
        StateManager::set_sp_value(con, &counter_keys[0], &(total + 1).to_spvalue()).await;
        StateManager::set_sp_value(con, &counter_keys[1], &(subsequent + 1).to_spvalue()).await;
    }

    StateManager::set_sp_value(
        con,
        &format!("{robot_name}_request_result"),
        &reason.to_spvalue(),
    )
    .await;
    StateManager::set_sp_value(
        con,
        &format!("{robot_name}_request_state"),
        &ActionRequestState::Failed.to_string().to_spvalue(),
    )
    .await;
}

/// Finish a request that was served.
///
/// The consecutive-failure streak is cleared here; `total_fail_counter` is
/// cumulative and is never reset.
pub async fn succeed_request(
    con: &mut SPConnection,
    robot_name: &str,
    result: &str,
    log_target: &str,
) {
    log::info!(target: log_target, "Request succeeded: {}.", result);

    StateManager::set_sp_value(
        con,
        &format!("{robot_name}_request_result"),
        &result.to_spvalue(),
    )
    .await;
    StateManager::set_sp_value(
        con,
        &format!("{robot_name}_subsequent_fail_counter"),
        &0.to_spvalue(),
    )
    .await;
    StateManager::set_sp_value(
        con,
        &format!("{robot_name}_request_state"),
        &ActionRequestState::Succeeded.to_string().to_spvalue(),
    )
    .await;
}

pub async fn publish_request_feedback(con: &mut SPConnection, robot_name: &str, feedback: &str) {
    StateManager::set_sp_value(
        con,
        &format!("{robot_name}_request_feedback"),
        &feedback.to_spvalue(),
    )
    .await;
}
