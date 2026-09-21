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

/// What colour to draw this robot, from `MESH_COLOR`.
///
/// `r,g,b` or `r,g,b,a`, each 0..1 - so `1,0.45,0.1` is an ABB's orange. Unset means
/// no colour: the meshes are drawn with their own materials, which is what a mesh
/// exported with materials wants. Set it for a robot whose meshes carry none, or the
/// viewer has nothing to go on and falls back to grey.
fn mesh_color(log_target: &str) -> Option<[f32; 4]> {
    let raw = std::env::var("MESH_COLOR").ok()?;
    let color = parse_mesh_color(&raw, log_target)?;
    log::info!(target: log_target, "Drawing the meshes in rgba {:?}.", color);
    Some(color)
}

/// `MESH_COLOR`'s value as an RGBA, or `None` if it does not describe one.
pub fn parse_mesh_color(raw: &str, log_target: &str) -> Option<[f32; 4]> {
    let parts: Result<Vec<f32>, _> = raw.split(',').map(|v| v.trim().parse::<f32>()).collect();
    let values = match parts {
        Ok(values) if values.len() == 3 || values.len() == 4 => values,
        Ok(values) => {
            log::warn!(
                target: log_target,
                "MESH_COLOR needs 3 or 4 comma-separated numbers, got {}; \
                 drawing the meshes with their own materials instead.",
                values.len()
            );
            return None;
        }
        Err(e) => {
            log::warn!(
                target: log_target,
                "MESH_COLOR is not a comma-separated list of numbers ({}); \
                 drawing the meshes with their own materials instead.",
                e
            );
            return None;
        }
    };

    // Alpha defaults to opaque rather than to zero: a colour asked for and then
    // drawn fully transparent is never what was meant.
    let color = [
        values[0],
        values[1],
        values[2],
        values.get(3).copied().unwrap_or(1.0),
    ];
    if color.iter().any(|c| !(0.0..=1.0).contains(c)) {
        log::warn!(
            target: log_target,
            "MESH_COLOR components are 0..1, got {:?}; using the meshes' own materials.",
            color
        );
        return None;
    }
    Some(color)
}

#[cfg(test)]
mod mesh_color_tests {
    use super::parse_mesh_color;

    #[test]
    fn three_components_are_opaque() {
        // Alpha defaults to 1, not 0: a colour asked for and drawn fully
        // transparent is never what was meant, and 0 is what this used to publish.
        assert_eq!(
            parse_mesh_color("0.961,0.486,0.125", "test"),
            Some([0.961, 0.486, 0.125, 1.0])
        );
    }

    #[test]
    fn four_components_keep_their_alpha() {
        assert_eq!(
            parse_mesh_color("1, 0.45, 0.1, 0.5", "test"),
            Some([1.0, 0.45, 0.1, 0.5])
        );
    }

    #[test]
    fn anything_that_is_not_a_colour_falls_back_to_the_meshes() {
        // Every one of these means "draw the meshes with their own materials",
        // which is the safe answer - a robot in the wrong colour is a bug you see,
        // a robot that fails to start is a cell that does not run.
        for raw in ["", "1,0", "1,0,0,0,0", "orange", "1,0,green", "2,0,0", "-1,0,0"] {
            assert_eq!(parse_mesh_color(raw, "test"), None, "{raw:?} is not a colour");
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
        mesh_color: mesh_color(&log_target),
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
