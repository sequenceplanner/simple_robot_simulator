use r2r;
use r2r::simple_robot_simulator_msgs::action::SimpleRobotControl;
use r2r::sensor_msgs::msg::JointState;

pub static NODE_ID: &'static str = "srs_event_based_test";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, NODE_ID, "")?;

    let client = node.create_action_client::<SimpleRobotControl::Action>("simple_robot_control")?;

    let waiting_for_server = node.is_available(&client)?;

    let handle = std::thread::spawn(move || loop {
        let _ = &node.spin_once(std::time::Duration::from_millis(100));
    });

    r2r::log_warn!(NODE_ID, "Waiting for Simple Robot Control Service...");
    waiting_for_server.await?;
    r2r::log_info!(NODE_ID, "Simple Robot Control Service available.");
    r2r::log_info!(NODE_ID, "Node started.");

    let goal = SimpleRobotControl::Goal {
        use_joint_positions: true,
        joint_positions: JointState {
            position: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        } 
        ..Default::default()
    };

    execute_event_based_command(goal, &client).await;

    handle.join().unwrap();

    Ok(())
}

async fn execute_event_based_command(
    goal: SimpleRobotControl::Goal,
    client: &r2r::ActionClient<SimpleRobotControl::Action>,
) -> bool {
    r2r::log_info!(NODE_ID, "Sending request to Simple Robot Simulator.");

    let (_goal, result, _feedback) = match client.send_goal_request(goal) {
        Ok(x) => match x.await {
            Ok(y) => y,
            Err(_) => {
                r2r::log_info!(NODE_ID, "Could not send goal request.");
                return false;
            }
        },
        Err(_) => {
            r2r::log_info!(NODE_ID, "Did not get goal.");
            return false;
        }
    };

    match result.await {
        Ok((status, msg)) => match status {
            r2r::GoalStatus::Aborted => {
                r2r::log_info!(NODE_ID, "Goal succesfully aborted with: {:?}", msg);
                true
            }
            _ => {
                r2r::log_info!(NODE_ID, "Executing the Simple Robot Simulator Command succeeded.");
                true
            }
        },
        Err(e) => {
            r2r::log_error!(NODE_ID, "Simple Robot Simulator Action failed with: {:?}", e,);
            false
        }
    }
}
