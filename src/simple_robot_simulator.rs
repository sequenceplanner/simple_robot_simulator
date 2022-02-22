use futures::stream::{Stream, StreamExt};
use k::nalgebra::Quaternion;
use k::prelude::InverseKinematicsSolver;
use k::{Chain, Node};
use k::{Isometry3, Translation3, UnitQuaternion, Vector3};
use r2r::geometry_msgs::msg::TransformStamped;
use r2r::sensor_msgs::msg::JointState;
use r2r::simple_robot_simulator_msgs::action::SimpleRobotControl;
use r2r::std_msgs::msg::Header;
use r2r::tf_tools_msgs::srv::LookupTransform;
use r2r::ActionServerGoal;
use r2r::ParameterValue;
use r2r::QosProfile;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;
use tokio::time::{sleep, Duration};

pub static NODE_ID: &'static str = "simple_robot_simulator";
pub static SIM_RATE_MS: u64 = 10;

#[derive(Default)]
pub struct Parameters {
    pub acceleration: f64,
    pub velocity: f64,
    // pub manual_control: bool // this is when controling with the gui to save poses etc...
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, NODE_ID, "")?;

    let params = node.params.clone();

    let params_things = params.lock().unwrap(); // OK to panic
    let params_things_clone_1 = params_things.clone();
    let params_things_clone_2 = params_things.clone();
    let params_things_clone_3 = params_things.clone();
    let params_things_clone_4 = params_things.clone();
    let use_urdf_from_path = params_things_clone_1.get("use_urdf_from_path");
    let urdf_path = params_things_clone_2.get("urdf_path");
    let urdf_raw = params_things_clone_3.get("urdf_raw");
    let initial_joint_state = params_things_clone_4.get("initial_joint_state");

    let (chain, joints, links) = match use_urdf_from_path {
        Some(p) => match p {
            ParameterValue::Bool(value) => match value {
                true => chain_from_urdf_path(urdf_path).await,
                false => {
                    match urdf_raw {
                        Some(p2) => match p2 {
                            ParameterValue::String(urdf) => chain_from_urdf_raw(urdf).await,
                            _ => {
                                r2r::log_error!(
                                    NODE_ID,
                                    "Parameter 'urdf_raw' has to be of type String."
                                );
                                panic!() // OK to panic, makes no sense to sontinue without a urdf.
                            }
                        },
                        None => {
                            r2r::log_error!(NODE_ID, "Parameter 'urdf_raw' not specified.");
                            panic!() // OK to panic, makes no sense to sontinue without a urdf.
                        }
                    }
                }
            },
            _ => {
                r2r::log_error!(
                    NODE_ID,
                    "Parameter 'use_urdf_from_path' has to be of type Bool."
                );
                panic!() // OK to panic, wrong type supplied.
            }
        },
        // Assumes use_urdf_from_path == true
        None => chain_from_urdf_path(urdf_path).await,
    };

    r2r::log_info!(NODE_ID, "Found joints: {:?}", joints);
    r2r::log_info!(NODE_ID, "Found links: {:?}", links);

    let act_joint_state = Arc::new(Mutex::new(JointState {
        header: Header {
            ..Default::default()
        },
        name: joints.clone(),
        position: match initial_joint_state {
            Some(p) => match p {
                ParameterValue::StringArray(joints) => joints
                    .iter()
                    .map(|j| j.parse::<f64>().unwrap_or_default())
                    .collect(),
                _ => {
                    r2r::log_warn!(
                        NODE_ID,
                        "Parameter 'initial_joint_state' has to be of type StringArray."
                    );
                    vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                }
            },
            None => {
                r2r::log_warn!(NODE_ID, "Parameter 'initial_joint_state' not specified.");
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
        },
        ..Default::default()
    }));

    let act_joint_state_clone_1 = act_joint_state.clone();
    let act_joint_state_clone_2 = act_joint_state.clone();

    let ref_joint_state = Arc::new(Mutex::new(JointState {
        header: Header {
            ..Default::default()
        },
        name: joints,
        ..Default::default()
    }));

    let ref_joint_state_clone_1 = ref_joint_state.clone();

    let ref_parameters = Arc::new(Mutex::new(Parameters::default()));
    let ref_parameters_clone_1 = ref_parameters.clone();

    let action_server =
        node.create_action_server::<SimpleRobotControl::Action>("simple_robot_control")?;
    let pub_timer = node.create_wall_timer(std::time::Duration::from_millis(SIM_RATE_MS))?;
    let joint_state_publisher =
        node.create_publisher::<JointState>("joint_states", QosProfile::default())?;

    tokio::task::spawn(async move {
        match publisher_callback(joint_state_publisher, pub_timer, &act_joint_state_clone_2).await {
            Ok(()) => (),
            Err(e) => r2r::log_error!(NODE_ID, "Joint state publisher failed with: '{}'.", e),
        };
    });

    let tf_lookup_client = node.create_client::<LookupTransform::Service>("tf_lookup")?;
    let waiting_for_tf_lookup_server = node.is_available(&tf_lookup_client)?;

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });

    r2r::log_warn!(NODE_ID, "Waiting for tf Lookup service...");
    waiting_for_tf_lookup_server.await?;
    r2r::log_info!(NODE_ID, "tf Lookup Service available.");
    r2r::log_info!(NODE_ID, "Simple Robot Simulator node started.");

    tokio::task::spawn(async move {
        let result = simple_robot_simulator_server(
            action_server,
            tf_lookup_client,
            chain,
            &act_joint_state_clone_1,
            &ref_joint_state_clone_1,
            &ref_parameters_clone_1,
        )
        .await;
        match result {
            Ok(()) => r2r::log_info!(NODE_ID, "Simple Robot Control Service call succeeded."),
            Err(e) => r2r::log_error!(
                NODE_ID,
                "Simple Robot Control Service call failed with: '{}'.",
                e
            ),
        };
    });

    handle.join().unwrap();

    Ok(())
}

async fn chain_from_urdf_path(
    urdf_path: Option<&ParameterValue>,
) -> (Chain<f64>, Vec<String>, Vec<String>) {
    let path = match urdf_path {
        Some(param) => match param {
            ParameterValue::String(value) => value,
            _ => {
                r2r::log_error!(NODE_ID, "Parameter 'urdf_path' has to be of type String.");
                panic!() // OK to panic, makes no sense to sontinue without a urdf.
            }
        },
        None => {
            r2r::log_error!(NODE_ID, "Parameter 'urdf_path' not specified.");
            panic!() // OK to panic, makes no sense to sontinue without a urdf.
        }
    };

    make_chain(path).await
}

async fn chain_from_urdf_raw(urdf: &str) -> (Chain<f64>, Vec<String>, Vec<String>) {
    let dir = match tempdir() {
        Ok(d) => d,
        Err(e) => {
            r2r::log_error!(
                NODE_ID,
                "Failed to generate temporary urdf directory with: '{}'.",
                e
            );
            panic!() // OK to panic, makes no sense to sontinue without a urdf.
        }
    };

    let urdf_path = dir.path().join("temp_urdf.urdf");
    let mut file = match File::create(urdf_path.clone()) {
        Ok(f) => {
            r2r::log_info!(NODE_ID, "Generated temporary urdf file at: {:?}", urdf_path);
            f
        }
        Err(e) => {
            r2r::log_error!(
                NODE_ID,
                "Failed to generate temporary urdf file with: '{}'.",
                e
            );
            panic!() // OK to panic, makes no sense to sontinue without a urdf.
        }
    };

    match write!(file, "{}", urdf) {
        Ok(()) => (),
        Err(e) => {
            r2r::log_error!(
                NODE_ID,
                "Failed to write to the temporary urdf file with: '{}'.",
                e
            );
            panic!() // OK to panic, makes no sense to sontinue without a urdf.
        }
    };

    let (c, j, l) = make_chain(match urdf_path.to_str() {
        Some(s) => s,
        None => {
            r2r::log_error!(NODE_ID, "Failed to convert path to string slice.");
            panic!()
        }
    })
    .await;

    drop(file);

    match dir.close() {
        Ok(()) => (),
        Err(e) => {
            r2r::log_error!(
                NODE_ID,
                "Failed to close and remove the temporary urdf directory with: '{}'.",
                e
            );
        }
    };

    (c, j, l)
}

async fn make_chain(urdf_path: &str) -> (Chain<f64>, Vec<String>, Vec<String>) {
    match k::Chain::<f64>::from_urdf_file(urdf_path.clone()) {
        Ok(c) => {
            r2r::log_info!(NODE_ID, "Loading urdf file: '{:?}'.", urdf_path);
            (
                c.clone(),
                c.iter_joints()
                    .map(|j| j.name.clone())
                    .collect::<Vec<String>>(),
                c.iter_links()
                    .map(|l| l.name.clone())
                    .collect::<Vec<String>>(),
            )
        }
        Err(e) => {
            r2r::log_error!(NODE_ID, "Failed to handle urdf with: '{}'.", e);
            panic!() // Still OK to panic, makes no sense to sontinue without a urdf.
        }
    }
}

async fn simple_robot_simulator_server(
    mut requests: impl Stream<Item = r2r::ActionServerGoalRequest<SimpleRobotControl::Action>> + Unpin,
    tf_lookup_client: r2r::Client<LookupTransform::Service>,
    chain: Chain<f64>,
    act_joint_state: &Arc<Mutex<JointState>>,
    ref_joint_state: &Arc<Mutex<JointState>>,
    ref_parameters: &Arc<Mutex<Parameters>>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        match requests.next().await {
            Some(request) => {
                let (mut g, mut _cancel) = match request.accept() {
                    Ok(res) => res,
                    Err(e) => {
                        r2r::log_error!(NODE_ID, "Could not accept goal request with '{}'.", e);
                        continue;
                    }
                };
                let g_clone = g.clone();
                match update_ref_values(
                    g_clone,
                    &chain,
                    &tf_lookup_client,
                    &act_joint_state,
                    &ref_joint_state,
                    &ref_parameters,
                )
                .await
                {
                    Some(()) => {
                        match simulate_movement_event_based(
                            &act_joint_state,
                            &ref_joint_state,
                            &ref_parameters,
                        )
                        .await
                        {
                            Some(()) => {
                                match g.succeed(SimpleRobotControl::Result { success: true }) {
                                    Ok(_) => (),
                                    Err(e) => {
                                        r2r::log_error!(
                                            NODE_ID,
                                            "Could not send result with '{}'.",
                                            e
                                        );
                                        continue;
                                    }
                                };
                                continue;
                            }
                            None => {
                                r2r::log_error!(
                                    NODE_ID,
                                    "Failed to simulate robot movement (event based).",
                                );
                                match g.abort(SimpleRobotControl::Result { success: false }) {
                                    Ok(_) => (),
                                    Err(e) => {
                                        r2r::log_error!(NODE_ID, "Failed to abort with '{}'.", e);
                                        continue;
                                    }
                                };
                                continue;
                            }
                        }
                    }
                    None => {
                        r2r::log_error!(NODE_ID, "Failed to update ref values.",);
                        match g.abort(SimpleRobotControl::Result { success: false }) {
                            Ok(_) => (),
                            Err(e) => {
                                r2r::log_error!(NODE_ID, "Failed to abort with '{}'.", e);
                                continue;
                            }
                        };
                        continue;
                    }
                };
            }
            None => (),
        }
    }
}

async fn update_ref_values(
    g: ActionServerGoal<SimpleRobotControl::Action>,
    chain: &Chain<f64>,
    tf_lookup_client: &r2r::Client<LookupTransform::Service>,
    act_joint_state: &Arc<Mutex<JointState>>,
    ref_joint_state: &Arc<Mutex<JointState>>,
    ref_parameters: &Arc<Mutex<Parameters>>,
) -> Option<()> {
    *ref_parameters.lock().unwrap() = Parameters {
        velocity: g.goal.velocity,
        acceleration: g.goal.acceleration,
    };

    let mut new_ref_joint_state = ref_joint_state.lock().unwrap().clone();

    match g.goal.use_joint_positions {
        true => {
            new_ref_joint_state.position = g.goal.joint_positions;
            *ref_joint_state.lock().unwrap() = new_ref_joint_state;
            Some(())
        }
        false => {
            let target_in_base = lookup_tf(
                &g.goal.base_frame_id,
                &g.goal.goal_feature_id,
                g.goal.tf_lookup_deadline,
                tf_lookup_client,
            )
            .await;

            let tcp_in_face_plate = lookup_tf(
                &g.goal.face_plate_id,
                &g.goal.tcp_id,
                g.goal.tf_lookup_deadline,
                tf_lookup_client,
            )
            .await;

            // add logging and error handling
            match tcp_in_face_plate {
                Some(tcp_frame) => match target_in_base {
                    Some(target_frame) => {
                        match generate_new_kinematic_chain(
                            &chain,
                            &g.goal.face_plate_id,
                            &g.goal.tcp_id,
                            &tcp_frame,
                        )
                        .await
                        {
                            Some(new_chain) => {
                                match calculate_inverse_kinematics(
                                    &new_chain,
                                    &g.goal.face_plate_id,
                                    &g.goal.tcp_id,
                                    &target_frame,
                                    &act_joint_state,
                                )
                                .await
                                {
                                    Some(new_joints) => {
                                        new_ref_joint_state.position = new_joints;
                                        *ref_joint_state.lock().unwrap() = new_ref_joint_state;
                                        Some(())
                                    }
                                    None => {
                                        r2r::log_error!(
                                            NODE_ID,
                                            "Failed to calculate inverse kinematics.",
                                        );
                                        None
                                    }
                                }
                            }
                            None => {
                                r2r::log_error!(NODE_ID, "Failed to generate new kinematic chain.",);
                                None
                            }
                        }
                    }
                    None => {
                        r2r::log_error!(
                            NODE_ID,
                            "Failed to lookup TF for: '{}' => '{}' in {} ms.",
                            &g.goal.base_frame_id,
                            &g.goal.goal_feature_id,
                            &g.goal.tf_lookup_deadline,
                        );
                        None
                    }
                },
                None => {
                    r2r::log_error!(
                        NODE_ID,
                        "Failed to lookup TF for: '{}' => '{}' in {} ms.",
                        &g.goal.face_plate_id,
                        &g.goal.tcp_id,
                        &g.goal.tf_lookup_deadline,
                    );
                    None
                }
            }
        }
    }
}

async fn generate_new_kinematic_chain(
    chain: &Chain<f64>,
    face_plate_id: &str,
    tcp_id: &str,
    frame: &TransformStamped,
) -> Option<Chain<f64>> {
    r2r::log_info!(NODE_ID, "Generating new kinematic chain.",);

    let face_plate_to_tcp_joint: Node<f64> = k::NodeBuilder::<f64>::new()
        .name(&format!("{}-{}", face_plate_id, tcp_id))
        .translation(Translation3::new(
            frame.transform.translation.x as f64,
            frame.transform.translation.y as f64,
            frame.transform.translation.z as f64,
        ))
        .rotation(UnitQuaternion::from_quaternion(Quaternion::new(
            frame.transform.rotation.w as f64,
            frame.transform.rotation.x as f64,
            frame.transform.rotation.y as f64,
            frame.transform.rotation.z as f64,
        )))
        .joint_type(k::JointType::Rotational {
            axis: Vector3::y_axis(),
        })
        .finalize()
        .into();

    let tcp_link = k::link::LinkBuilder::new().name(tcp_id).finalize();
    face_plate_to_tcp_joint.set_link(Some(tcp_link));

    match chain
        .iter_joints()
        .map(|j| j.name.clone())
        .collect::<Vec<String>>()
        .last()
    {
        Some(parent) => match chain.find(parent) {
            Some(parent_node) => {
                face_plate_to_tcp_joint.set_parent(parent_node);
                let mut new_chain_nodes: Vec<k::Node<f64>> =
                    chain.iter().map(|x| x.clone()).collect();
                new_chain_nodes.push(face_plate_to_tcp_joint);
                Some(Chain::from_nodes(new_chain_nodes))
            }
            None => {
                r2r::log_error!(NODE_ID, "Failed to set parent node.");
                None
            }
        },
        None => {
            r2r::log_error!(NODE_ID, "Failed to find parent node in the chain.");
            None
        }
    }
}

async fn calculate_inverse_kinematics(
    new_chain: &Chain<f64>,
    face_plate_id: &str,
    tcp_id: &str,
    target_frame: &TransformStamped,
    act_joint_state: &Arc<Mutex<JointState>>,
) -> Option<Vec<f64>> {
    r2r::log_info!(NODE_ID, "Calculating inverse kinematics.",);
    let mut pos = act_joint_state.lock().unwrap().clone().position;
    pos.push(0.0);

    match new_chain.find(&format!("{}-{}", face_plate_id, tcp_id)) {
        Some(ee_joint) => {
            let arm = k::SerialChain::from_end(ee_joint);
            let mut positions = act_joint_state.lock().unwrap().clone().position;
            positions.push(0.0);
            match arm.set_joint_positions(&positions) {
                Ok(()) => {
                    let solver = k::JacobianIkSolver::new(0.01, 0.01, 0.5, 50);

                    let target = Isometry3::from_parts(
                        Translation3::new(
                            target_frame.transform.translation.x as f64,
                            target_frame.transform.translation.y as f64,
                            target_frame.transform.translation.z as f64,
                        ),
                        UnitQuaternion::from_quaternion(Quaternion::new(
                            target_frame.transform.rotation.w as f64,
                            target_frame.transform.rotation.x as f64,
                            target_frame.transform.rotation.y as f64,
                            target_frame.transform.rotation.z as f64,
                        )),
                    );

                    r2r::log_info!(NODE_ID, "Solving inverse kinematics.",);

                    let constraints = k::Constraints {
                        ignored_joint_names: vec![format!("{}-{}", face_plate_id, tcp_id)],
                        ..Default::default()
                    };

                    match solver.solve_with_constraints(&arm, &target, &constraints) {
                        Ok(()) => {
                            let mut j = arm.joint_positions();
                            match j.pop() {
                                Some(_) => Some(j),
                                None => {
                                    r2r::log_error!(
                                        NODE_ID,
                                        "Failed to shring joint dof to original size.",
                                    );
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            r2r::log_error!(
                                NODE_ID,
                                "Failed to solve with constraints with: '{}'.",
                                e
                            );
                            None
                        }
                    }
                }
                Err(e) => {
                    r2r::log_error!(
                        NODE_ID,
                        "Failed to set joint positions for arm with: '{}'.",
                        e
                    );
                    None
                }
            }
        }
        None => None,
    }
}

async fn simulate_movement_event_based(
    act_joint_state: &Arc<Mutex<JointState>>,
    ref_joint_state: &Arc<Mutex<JointState>>,
    ref_parameters: &Arc<Mutex<Parameters>>,
) -> Option<()> {

    let test_velocity = ref_parameters.lock().unwrap().velocity;
    let velocity = if test_velocity == 0.0 {
        1.0
    } else {
        test_velocity
    };

    let act_values = act_joint_state.lock().unwrap().clone();
    let joint_names = act_values.name;

    let mut act_pos = act_values
        .position
        .iter()
        .map(|x| (*x * 10000.0).round() / 10000.0)
        .collect::<Vec<f64>>();

    let ref_pos = ref_joint_state
        .lock()
        .unwrap()
        .clone()
        .position
        .iter()
        .map(|x| (*x * 10000.0).round() / 10000.0)
        .collect::<Vec<f64>>();

    while act_pos != ref_pos {
        let ssc = ref_pos
            .iter()
            .zip(act_pos.iter())
            .map(|(x, y)| (x - y).abs())
            .collect::<Vec<f64>>();

        let ssc_max = ssc.iter().copied().fold(f64::NAN, f64::max);
        let sync_max_factor;
        if ssc_max == 0.0 {
            sync_max_factor = 1.0
        } else {
            sync_max_factor = 1.0 / ssc_max
        }

        let new_ssc = ssc
            .iter()
            .map(|x| x * sync_max_factor)
            .collect::<Vec<f64>>();

        let mut new_act_pos = Vec::<f64>::new();
        ref_pos
            .iter()
            .zip(act_pos.iter())
            .zip(new_ssc.iter())
            .for_each(|((x, y), z)| {
                let step = 0.1 * velocity * std::f64::consts::PI / 180.0;

                if x < &(y - step) && z > &0.1 {
                    new_act_pos.push(((y - step * z) * 10000.0).round() / 10000.0)
                } else if x > &(y + step) && z > &0.1 {
                    new_act_pos.push(((y + step * z) * 10000.0).round() / 10000.0)
                } else {
                    new_act_pos.push((*x * 10000.0).round() / 10000.0)
                }
            });

        act_pos = new_act_pos.clone();

        *act_joint_state.lock().unwrap() = JointState {
            header: Header {
                ..Default::default()
            },
            name: joint_names.clone(),
            position: new_act_pos.clone(),
            ..Default::default()
        };

        sleep(Duration::from_millis(SIM_RATE_MS)).await;
    }
    Some(())
}

// async fn simulate_movement_state_based() {
    // use regular pub sub...
// }

async fn publisher_callback(
    publisher: r2r::Publisher<JointState>,
    mut timer: r2r::Timer,
    joint_state: &Arc<Mutex<JointState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let mut clock = r2r::Clock::create(r2r::ClockType::RosTime).unwrap();
        let now = clock.get_now().unwrap();
        let time_stamp = r2r::Clock::to_builtin_time(&now);

        let position = joint_state.lock().unwrap().clone().position;

        let updated_joint_state = JointState {
            header: Header {
                stamp: time_stamp.clone(),
                ..Default::default()
            },
            name: joint_state.lock().unwrap().clone().name,
            position,
            ..Default::default()
        };
        
        match publisher.publish(&updated_joint_state) {
            Ok(()) => (),
            Err(e) => {
                r2r::log_error!(NODE_ID, "Publisher failed to send a message with: '{}'", e);
            }
        };
        timer.tick().await?;
    }
}

async fn lookup_tf(
    parent_id: &str,
    child_id: &str,
    deadline: i32,
    tf_lookup_client: &r2r::Client<LookupTransform::Service>,
) -> Option<TransformStamped> {
    let request = LookupTransform::Request {
        parent_id: parent_id.to_string(),
        child_id: child_id.to_string(),
        deadline,
    };

    let response = tf_lookup_client
        .request(&request)
        .expect("Could not send TF Lookup request.")
        .await
        .expect("Cancelled.");

    r2r::log_info!(
        NODE_ID,
        "Request to lookup parent '{}' to child '{}' sent.",
        parent_id,
        child_id
    );

    match response.success {
        true => Some(response.transform),
        false => {
            r2r::log_error!(
                NODE_ID,
                "Couldn't lookup tf for parent '{}' and child '{}'.",
                parent_id,
                child_id
            );
            None
        }
    }
}
