use micro_sp::*;

/// Seed every Redis key this simulator reads or writes.
///
/// The key names mirror `ur_redis_driver` exactly, including the ones a kinematic
/// simulator cannot honour (`blend_radius`, `payload`, `execution_time`, ...). They
/// are accepted and ignored so a runner can point at the simulator or the real arm
/// without changing which keys it writes.
///
/// Everything is seeded `UNKNOWN` except the flags and counters, which need a
/// concrete starting value. Two parameters are seeded concretely for the same reason
/// `ur_redis_driver` does it: `command_server` fails any request whose full key set
/// is not present, so a key it reads must exist before the first request arrives.
pub fn generate_robot_interface_state(robot_name: &str, log_target: &str) -> State {
    let state = State::new();

    // --- the request handshake ---
    let request_trigger = bv!(&&format!("{}_request_trigger", robot_name));
    let request_state = v!(&&format!("{}_request_state", robot_name));
    let request_cancel = bv!(&&format!("{}_request_cancel", robot_name));
    let request_result = v!(&&format!("{}_request_result", robot_name));
    let request_feedback = v!(&&format!("{}_request_feedback", robot_name));
    let total_fail_counter = iv!(&&format!("{}_total_fail_counter", robot_name));
    let subsequent_fail_counter = iv!(&&format!("{}_subsequent_fail_counter", robot_name));

    let state = state.add(assign!(request_trigger, false.to_spvalue()), &log_target);
    let state = state.add(assign!(request_state, "initial".to_spvalue()), &log_target);
    let state = state.add(assign!(request_cancel, false.to_spvalue()), &log_target);
    let state = state.add(
        assign!(request_result, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(request_feedback, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(assign!(total_fail_counter, 0.to_spvalue()), &log_target);
    let state = state.add(assign!(subsequent_fail_counter, 0.to_spvalue()), &log_target);

    // --- request parameters ---
    let command_type = v!(&&format!("{}_command_type", robot_name));
    let acceleration = fv!(&&format!("{}_acceleration", robot_name));
    let velocity = fv!(&&format!("{}_velocity", robot_name));
    let global_acceleration_scaling = fv!(&&format!("{}_global_acceleration_scaling", robot_name));
    let global_velocity_scaling = fv!(&&format!("{}_global_velocity_scaling", robot_name));
    let use_execution_time = bv!(&&format!("{}_use_execution_time", robot_name));
    let execution_time = fv!(&&format!("{}_execution_time", robot_name));
    let use_blend_radius = bv!(&&format!("{}_use_blend_radius", robot_name));
    let blend_radius = fv!(&&format!("{}_blend_radius", robot_name));
    let use_joint_positions = bv!(&&format!("{}_use_joint_positions", robot_name));
    let joint_positions = av!(&&format!("{}_joint_positions", robot_name));
    let use_preferred_joint_config = bv!(&&format!("{}_use_preferred_joint_config", robot_name));
    let preferred_joint_config = av!(&&format!("{}_preferred_joint_config", robot_name));
    let use_payload = bv!(&&format!("{}_use_payload", robot_name));
    let payload = v!(&&format!("{}_payload", robot_name));
    let baseframe_id = v!(&&format!("{}_baseframe_id", robot_name));
    let faceplate_id = v!(&&format!("{}_faceplate_id", robot_name));
    let goal_feature_id = v!(&&format!("{}_goal_feature_id", robot_name));
    let tcp_id = v!(&&format!("{}_tcp_id", robot_name));
    let root_frame_id = v!(&&format!("{}_root_frame_id", robot_name));
    let use_relative_pose = bv!(&&format!("{}_use_relative_pose", robot_name));
    let relative_pose = av!(&&format!("{}_relative_pose", robot_name));
    let force_threshold = fv!(&&format!("{}_force_threshold", robot_name));

    let state = state.add(
        assign!(command_type, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(acceleration, SPValue::Float64(FloatOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(velocity, SPValue::Float64(FloatOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(global_acceleration_scaling, SPValue::Float64(FloatOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(global_velocity_scaling, SPValue::Float64(FloatOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(use_execution_time, SPValue::Bool(BoolOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(execution_time, SPValue::Float64(FloatOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(use_blend_radius, SPValue::Bool(BoolOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(blend_radius, SPValue::Float64(FloatOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(use_joint_positions, SPValue::Bool(BoolOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(joint_positions, SPValue::Array(ArrayOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(use_preferred_joint_config, SPValue::Bool(BoolOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(preferred_joint_config, SPValue::Array(ArrayOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(use_payload, SPValue::Bool(BoolOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(payload, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(baseframe_id, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(faceplate_id, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(goal_feature_id, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(tcp_id, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(root_frame_id, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(use_relative_pose, SPValue::Bool(BoolOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(relative_pose, SPValue::Array(ArrayOrUnknown::UNKNOWN)),
        &log_target,
    );
    let state = state.add(
        assign!(force_threshold, SPValue::Float64(FloatOrUnknown::UNKNOWN)),
        &log_target,
    );

    // --- published state ---
    // Seeded UNKNOWN; the state publisher writes the real vector on its first tick.
    let joint_states = av!(&&format!("{}_joint_states", robot_name));
    let robot_connected = bv!(&&format!("{}_robot_connected", robot_name));
    let robot_model = v!(&&format!("{}_robot_model", robot_name));

    let state = state.add(
        assign!(joint_states, SPValue::Array(ArrayOrUnknown::UNKNOWN)),
        &log_target,
    );
    // A simulated robot is reachable as soon as the process is up, and the runner
    // gates motion on this the same way it does for the real arm.
    let state = state.add(assign!(robot_connected, true.to_spvalue()), &log_target);
    let state = state.add(
        assign!(robot_model, SPValue::String(StringOrUnknown::UNKNOWN)),
        &log_target,
    );

    state
}
