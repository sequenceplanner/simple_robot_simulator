use micro_sp::{
    ConnectionManager, DEFAULT_HEALTH_CHECK_PERIOD, StateManager, initialize_env_logger,
};
use simple_robot_simulator::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// An environment variable, warning and falling back rather than aborting - a typo
/// in a launcher's .env should not take the simulator down.
fn env_or(name: &str, default: &str, log_target: &str) -> String {
    match std::env::var(name) {
        Ok(value) => value,
        Err(_) => {
            log::warn!(target: log_target, "{} is not set, using '{}'.", name, default);
            default.to_string()
        }
    }
}

/// The joint configuration the robot starts in.
///
/// `INITIAL_JOINT_STATE` is a comma-separated list, replacing the ROS parameter of
/// the same name. Without it each joint starts at the midpoint of its limits, which
/// is what the ROS node did, and a joint with no limits starts at zero.
fn initial_joint_positions(description: &RobotDescription, log_target: &str) -> Vec<f64> {
    let from_limits = || -> Vec<f64> {
        description
            .chain
            .iter_joints()
            .map(|joint| match joint.limits {
                Some(limits) => (limits.max + limits.min) / 2.0,
                None => 0.0,
            })
            .collect()
    };

    let Ok(raw) = std::env::var("INITIAL_JOINT_STATE") else {
        log::info!(
            target: log_target,
            "INITIAL_JOINT_STATE is not set, starting at the midpoint of each joint's limits."
        );
        return from_limits();
    };

    let parsed: Result<Vec<f64>, _> = raw
        .split(',')
        .map(|value| value.trim().parse::<f64>())
        .collect();

    match parsed {
        Ok(positions) if positions.len() == description.joints.len() => positions,
        Ok(positions) => {
            log::warn!(
                target: log_target,
                "INITIAL_JOINT_STATE has {} values but this robot has {} joints, \
                 using the midpoint of each joint's limits instead.",
                positions.len(),
                description.joints.len()
            );
            from_limits()
        }
        Err(e) => {
            log::warn!(
                target: log_target,
                "INITIAL_JOINT_STATE is not a comma-separated list of numbers ({}), \
                 using the midpoint of each joint's limits instead.",
                e
            );
            from_limits()
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    initialize_env_logger();

    let robot_id = env_or("ROBOT_ID", "r1", "simple_robot_simulator");
    let log_target = format!("{}_simple_robot_simulator", robot_id);

    let robot_model = env_or("ROBOT_MODEL", "ur10e", &log_target);
    let description_dir = env_or("DESCRIPTION_DIR", "description/", &log_target);

    // Namespace for the frames this simulator publishes. Empty by default, and empty
    // is exactly the identity, so a single-robot cell keeps publishing the bare
    // `base_link`..`tool0` its scene files already name. Deliberately not derived
    // from ROBOT_ID: opt-in only.
    let tf_prefix = std::env::var("SIM_TF_PREFIX").unwrap_or_default();
    if !tf_prefix.is_empty() {
        log::info!(target: &log_target, "Publishing frames under the prefix '{}'.", tf_prefix);
    }

    let mut urdf_path = PathBuf::from(&description_dir);
    urdf_path.push(format!("urdf/{}.urdf", robot_model));
    let mut meshes_path = PathBuf::from(&description_dir);
    meshes_path.push("meshes");
    meshes_path.push(&robot_model);
    meshes_path.push("visual");

    let params = URDFParameters {
        name: robot_id.clone(),
        tf_prefix: tf_prefix.clone(),
        robot_model: robot_model.clone(),
        description_file: urdf_path.to_string_lossy().to_string(),
        meshes_path: meshes_path.to_string_lossy().to_string(),
    };

    // Without a URDF there is no robot to simulate, so this is a real startup
    // failure - but it says which file it could not load.
    let description = Arc::new(load_description(&params.description_file)?);
    log::info!(target: &log_target, "Loaded '{}'.", params.description_file);
    log::info!(target: &log_target, "Found joints: {:?}", description.joints);
    log::info!(target: &log_target, "Found links: {:?}", description.links);

    let initial = initial_joint_positions(&description, &log_target);
    log::info!(target: &log_target, "Starting at: {:?}", initial);

    let state = generate_robot_interface_state(&robot_id, &log_target);

    // The `Arc` has to exist before the health monitor can be spawned - it takes
    // `self: &Arc<Self>` so the background task can hold its own reference.
    let connection_manager = Arc::new(ConnectionManager::new().await);
    let mut con = connection_manager.connection();
    StateManager::set_state(&mut con, &state).await;

    // One PING every few seconds for the whole process, purely so an unreachable
    // Redis shows up in the log. Nothing depends on it to recover: the handles the
    // tasks hold reconnect themselves.
    connection_manager.spawn_health_monitor(&log_target, DEFAULT_HEALTH_CHECK_PERIOD);

    let simulator_state = Arc::new(Mutex::new(SimulatorState::new(initial)));

    // Its own chain, deliberately not a clone of the publisher's: see `load_chain`.
    // Sharing one would have the state publisher re-posing the arm underneath every
    // IK solve.
    let ik_chain = load_chain(&params.description_file)?;

    let command_server_task = command_server(
        &robot_id,
        &tf_prefix,
        ik_chain,
        &connection_manager,
        simulator_state.clone(),
    );

    let state_publisher_task = state_publisher(
        params.clone(),
        description.clone(),
        &connection_manager,
        simulator_state.clone(),
    );

    log::info!(target: &log_target, "Simple Robot Simulator online.");

    tokio::try_join!(command_server_task, state_publisher_task)?;

    Ok(())
}

#[tokio::main]
async fn main() {
    loop {
        if let Err(e) = run().await {
            log::error!(target: "simple_robot_simulator", "Fatal error: {}", e);
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
}
