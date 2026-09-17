//! Drive the simulator from the command line, the way a runner would.
//!
//! Replaces the two r2r action-client test binaries. It writes the request keys,
//! sets the trigger and polls `request_state` until it reaches a terminal value.
//!
//! ```text
//! cargo run --bin sim_test_client -- frame pos1 svt_tcp
//! cargo run --bin sim_test_client -- joints 0.0,-1.5707,1.5707,-1.5707,-1.5707,0.0
//! cargo run --bin sim_test_client -- cancel
//! cargo run --bin sim_test_client -- lookup base_link svt_tcp
//! cargo run --bin sim_test_client -- capture pos1 base_link svt_tcp
//! ```

use micro_sp::*;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_env_logger();

    let robot_id = std::env::var("ROBOT_ID").unwrap_or_else(|_| "r1".to_string());
    let log_target = format!("{}_sim_test_client", robot_id);
    let args: Vec<String> = std::env::args().skip(1).collect();

    let connection_manager = ConnectionManager::new().await;
    let mut con = connection_manager.get_connection().await;

    let key = |suffix: &str| format!("{robot_id}_{suffix}");
    let set = async |con: &mut SPConnection, k: String, v: SPValue| {
        StateManager::set_sp_value(con, &k, &v).await;
    };

    let mode = args.first().map(|s| s.as_str()).unwrap_or("frame");

    // Resolving a frame pair is the quickest way to check the arm actually ended up
    // where the goal said, rather than merely reporting that it did.
    if mode == "lookup" {
        let parent = args.get(1).cloned().unwrap_or_else(|| "base_link".to_string());
        let child = args.get(2).cloned().unwrap_or_else(|| "svt_tcp".to_string());
        let tf = TransformsManager::lookup_transform(&mut con, &parent, &child).await?;
        println!(
            "{} -> {}: x={:.4} y={:.4} z={:.4}",
            parent,
            child,
            tf.transform.translation.x.into_inner(),
            tf.transform.translation.y.into_inner(),
            tf.transform.translation.z.into_inner()
        );
        return Ok(());
    }

    // Teach a waypoint from where the arm is standing: resolve a frame pair and store
    // the result as a frame of its own, which is then a goal the arm can be sent back
    // to.
    if mode == "capture" {
        let name = args.get(1).cloned().ok_or("usage: capture <name> [parent] [child]")?;
        let parent = args.get(2).cloned().unwrap_or_else(|| "base_link".to_string());
        let child = args.get(3).cloned().unwrap_or_else(|| "svt_tcp".to_string());

        let mut tf = TransformsManager::lookup_transform(&mut con, &parent, &child).await?;
        tf.parent_frame_id = parent.clone();
        tf.child_frame_id = name.clone();
        tf.active_transform = false;
        tf.time_stamp = std::time::SystemTime::now();
        TransformsManager::insert_transform(&mut con, &tf).await?;

        println!(
            "captured '{}' under '{}' at x={:.4} y={:.4} z={:.4}",
            name,
            parent,
            tf.transform.translation.x.into_inner(),
            tf.transform.translation.y.into_inner(),
            tf.transform.translation.z.into_inner()
        );
        return Ok(());
    }

    if mode == "cancel" {
        set(&mut con, key("request_cancel"), true.to_spvalue()).await;
        println!("cancel requested");
        return Ok(());
    }

    let velocity: f64 = std::env::var("VELOCITY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);

    set(&mut con, key("command_type"), "move_j".to_spvalue()).await;
    set(&mut con, key("velocity"), velocity.to_spvalue()).await;
    set(&mut con, key("acceleration"), 0.5.to_spvalue()).await;

    match mode {
        "joints" => {
            let raw = args.get(1).ok_or("usage: sim_test_client joints <j1,j2,...>")?;
            let positions: Vec<f64> = raw
                .split(',')
                .map(|v| v.trim().parse::<f64>())
                .collect::<Result<_, _>>()?;
            println!("moving to joint positions {:?}", positions);

            set(&mut con, key("use_joint_positions"), true.to_spvalue()).await;
            set(
                &mut con,
                key("joint_positions"),
                SPValue::Array(ArrayOrUnknown::Array(
                    positions.iter().map(|p| p.to_spvalue()).collect(),
                )),
            )
            .await;
        }
        "frame" => {
            let goal_feature_id = args.get(1).cloned().unwrap_or_else(|| "pos1".to_string());
            let tcp_id = args.get(2).cloned().unwrap_or_else(|| "svt_tcp".to_string());
            println!("moving '{}' to frame '{}'", tcp_id, goal_feature_id);

            set(&mut con, key("use_joint_positions"), false.to_spvalue()).await;
            set(&mut con, key("baseframe_id"), "base_link".to_spvalue()).await;
            set(&mut con, key("faceplate_id"), "tool0".to_spvalue()).await;
            set(&mut con, key("goal_feature_id"), goal_feature_id.to_spvalue()).await;
            set(&mut con, key("tcp_id"), tcp_id.to_spvalue()).await;
        }
        other => return Err(format!("unknown mode '{}': use frame, joints or cancel", other).into()),
    }

    // The simulator only acts on a request whose state is `initial`, so reset it
    // before triggering.
    set(
        &mut con,
        key("request_state"),
        ActionRequestState::Initial.to_string().to_spvalue(),
    )
    .await;
    set(&mut con, key("request_trigger"), true.to_spvalue()).await;

    let started = Instant::now();
    let mut last = String::new();
    loop {
        tokio::time::sleep(Duration::from_millis(50)).await;

        let state = StateManager::get_sp_value(&mut con, &key("request_state"))
            .await
            .map(|v| v.to_string())
            .unwrap_or_else(|| "UNKNOWN".to_string());

        if state != last {
            println!("[{:>6.2}s] request_state: {}", started.elapsed().as_secs_f64(), state);
            last = state.clone();
        }

        if state == ActionRequestState::Succeeded.to_string()
            || state == ActionRequestState::Failed.to_string()
        {
            let result = StateManager::get_sp_value(&mut con, &key("request_result"))
                .await
                .map(|v| v.to_string())
                .unwrap_or_default();
            println!("request_result: {}", result);

            if let Some(joints) = StateManager::get_sp_value(&mut con, &key("joint_states")).await {
                println!("joint_states:   {}", joints);
            }

            let _ = log_target;
            return Ok(());
        }

        if started.elapsed() > Duration::from_secs(120) {
            return Err("timed out waiting for a terminal request state".into());
        }
    }
}
