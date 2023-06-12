use futures::stream::{Stream, StreamExt};
use k::nalgebra::Quaternion;
use k::prelude::InverseKinematicsSolver;
use k::{Chain, Node};
use k::{Isometry3, Translation3, UnitQuaternion, Vector3};
use r2r::geometry_msgs::msg::TransformStamped;
use r2r::sensor_msgs::msg::JointState;
use r2r::simple_robot_simulator_msgs::action::SimpleRobotControl;
use r2r::std_msgs::msg::Header;
use r2r::std_srvs::srv::{SetBool, Trigger};
use r2r::scene_manipulation_msgs::srv::LookupTransform;
use r2r::ActionServerGoal;
use r2r::ParameterValue;
use r2r::QosProfile;
use r2r::ServiceRequest;
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // setup the node
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, NODE_ID, "")?;

        // handle parameters passed on from the launch filessimple_robot_simulator
        let params = node.params.clone();
        let params_things = params.lock().unwrap(); // OK to panic
        let urdf_raw = params_things.get("urdf_raw");
        let initial_joint_state = params_things.get("initial_joint_state");
        let prefix = params_things.get("prefix");

        // make a manipulatable kinematic chain using a urdf or through the xacro pipeline
        let (chain, joints, links) = match urdf_raw {
            Some(p2) => match p2 {
                ParameterValue::String(urdf) => chain_from_urdf_raw(urdf).await,
                _ => {
                    r2r::log_error!(NODE_ID, "Parameter 'urdf_raw' has to be of type String.");
                    panic!() // OK to panic, makes no sense to continue without a urdf
                }
            },
            None => {
                r2r::log_error!(NODE_ID, "Parameter 'urdf_raw' not specified.");
                panic!() // OK to panic, makes no sense to continue without a urdf
            }
        };

        // did we get what we expected
        r2r::log_info!(NODE_ID, "Found joints: {:?}", joints);
        r2r::log_info!(NODE_ID, "Found links: {:?}", links);

    let initial_joint_value = JointState {
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
                    chain
                    .iter_joints()
                    .map(|_| 0.0)
                    .collect()
                }
            },
            None => {
                r2r::log_info!(
                    NODE_ID,
                    "Parameter 'initial_joint_state' not specified, avg of limits will be used."
                );
                chain
                    .iter_joints()
                    .map(|x| match x.limits {
                        Some(l) => (l.max + l.min) / 2.0,
                        None => 0.0,
                    })
                    .collect()
            }
        },
        ..Default::default()
    };

    // where is the robot now
    let act_joint_state= Arc::new(Mutex::new(initial_joint_value));

    // where does the robot have to go
    let ref_joint_state = Arc::new(Mutex::new(JointState {
        header: Header {
            ..Default::default()
        },
        name: joints.clone(),
        ..Default::default()
    }));

    // initialize the ghost joint state
    let ghost_joint_state = Arc::new(Mutex::new(JointState {
        header: Header {
            ..Default::default()
        },
        name: joints.clone(),
        ..Default::default()
    }));

    // keep track of the parameters
    let ref_parameters = Arc::new(Mutex::new(Parameters::default()));

    // do we enable direct joint state control through a subscriber (disabled)
    // or only the regular action request control (enabled - default)
    let remote_control = Arc::new(Mutex::new(true));

    // the main action service of this node to control the robot simulation
    let action_server =
        node.create_action_server::<SimpleRobotControl::Action>("simple_robot_control")?;

    // a service to enable or disable remote control
    let remote_control_service =
        node.create_service::<SetBool::Service>("enable_remote_control")?;

    // // a service to enable or disable teaching mode
    // let teaching_mode_service =
    // node.create_service::<SetBool::Service>("enable_teaching_mode")?;

    // a service to trigger the move from current pose to ghost pose
    let match_ghost_service =
        node.create_service::<Trigger::Service>("match_ghost")?;

    // listen to direct joint state control when remote control is disabled
    let joint_state_subscriber =
        node.subscribe::<JointState>("simple_joint_control", QosProfile::default())?;

    // listen to the current teaching ghost pose to go to (in teaching mode?)
    let ghost_state_subscriber =
        node.subscribe::<JointState>("ghost/joint_states", QosProfile::default())?;

    // publish the actual joint state of the robot at the simulation rate
    let pub_timer_1 = node.create_wall_timer(std::time::Duration::from_millis(SIM_RATE_MS))?;
    let joint_state_publisher =
        node.create_publisher::<JointState>("joint_states", QosProfile::default())?;

    // spawn a tokio task to handle publishing the joint state
    let act_joint_state_clone_1 = act_joint_state.clone();
    tokio::task::spawn(async move {
        match publisher_callback(joint_state_publisher, pub_timer_1, &act_joint_state_clone_1).await
        {
            Ok(()) => (),
            Err(e) => r2r::log_error!(NODE_ID, "Joint state publisher failed with: '{}'.", e),
        };
    });

    // spawn a tokio task to listen to incomming joint state messages
    let ref_joint_state_clone_1 = ref_joint_state.clone();
    let remote_control_clone_1 = remote_control.clone();
    tokio::task::spawn(async move {
        match joint_subscriber_callback(
            joint_state_subscriber,
            &ref_joint_state_clone_1,
            &remote_control_clone_1,
        )
        .await
        {
            Ok(()) => (),
            Err(e) => r2r::log_error!(NODE_ID, "Joint state subscriber failed with {}.", e),
        };
    });

    // spawn a tokio task to listen to incomming ghost joint state messages
    //(maybe only enable this in teaching mode to save computation)
    let ghost_joint_state_clone_1 = ghost_joint_state.clone();
    tokio::task::spawn(async move {
        match ghost_subscriber_callback(
            ghost_state_subscriber,
            &ghost_joint_state_clone_1
        )
        .await
        {
            Ok(()) => (),
            Err(e) => r2r::log_error!(NODE_ID, "Joint state subscriber failed with {}.", e),
        };
    });

    // a client that asks a tf lookup service for transformations between frames in the tf tree
    let tf_lookup_client = node.create_client::<LookupTransform::Service>("/lookup_transform")?;
    let waiting_for_tf_lookup_server = node.is_available(&tf_lookup_client)?;

    // keep the node alive
    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });

    // before the other things in this node can start, it makes sense to wait for the tf lookup service to become alive
    r2r::log_warn!(NODE_ID, "Waiting for tf Lookup service...");
    waiting_for_tf_lookup_server.await?;
    r2r::log_info!(NODE_ID, "tf Lookup Service available.");

    // offer a service to enable or disable remote control
    let remote_control_clone_2 = remote_control.clone();
    tokio::task::spawn(async move {
        let result = remote_control_server(remote_control_service, &remote_control_clone_2).await;
        match result {
            Ok(()) => r2r::log_info!(NODE_ID, "Remote Control Service call succeeded."),
            Err(e) => r2r::log_error!(NODE_ID, "Remote Control Service call failed with: {}.", e),
        };
    });

    // offer a service to trigger the movement from the current position to match the ghost
    let ref_joint_state_clone_3 = ref_joint_state.clone();
    let act_joint_state_clone_4 = act_joint_state.clone();
    let ghost_joint_state_clone_2 = ghost_joint_state.clone();
    let ref_parameters_clone_3 = ref_parameters.clone();
    tokio::task::spawn(async move {
        let result = match_ghost_server(match_ghost_service, &ref_joint_state_clone_3, &act_joint_state_clone_4, &ghost_joint_state_clone_2, &ref_parameters_clone_3).await;
        match result {
            Ok(()) => r2r::log_info!(NODE_ID, "Match Ghost Service call succeeded."),
            Err(e) => r2r::log_error!(NODE_ID, "Match Ghost Service call failed with: {}.", e),
        };
    });

    // spawn a tokio task that handles the main action service of this node
    let remote_control_clone_3 = remote_control.clone();
    let act_joint_state_clone_3 = act_joint_state.clone();
    let ref_joint_state_clone_2 = ref_joint_state.clone();
    let ref_parameters_clone_2 = ref_parameters.clone();
    tokio::task::spawn(async move {
        let result = simple_robot_simulator_server(
            action_server,
            tf_lookup_client,
            chain,
            &remote_control_clone_3,
            &act_joint_state_clone_3,
            &ref_joint_state_clone_2,
            &ref_parameters_clone_2,
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

    r2r::log_info!(NODE_ID, "Simple Robot Simulator node started.");

    handle.join().unwrap();

    Ok(())
}

// if going throught the launch file - xacro - robot description pipeline,
// a raw urdf is provided, which has to be put in a temp file for the
// kinematic chain generator to access (there might be a nicer way to do this)
async fn chain_from_urdf_raw(urdf: &str) -> (Chain<f64>, Vec<String>, Vec<String>) {
    // create the temp directory to store the urdf file in
    let dir = match tempdir() {
        Ok(d) => d,
        Err(e) => {
            r2r::log_error!(
                NODE_ID,
                "Failed to generate temporary urdf directory with: '{}'.",
                e
            );
            panic!() // OK to panic, makes no sense to continue without a urdf.
        }
    };

    // create the temporary urdf file
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
            panic!() // OK to panic, makes no sense to continue without a urdf.
        }
    };

    // dump the raw urdf to the generated file
    match write!(file, "{}", urdf) {
        Ok(()) => (),
        Err(e) => {
            r2r::log_error!(
                NODE_ID,
                "Failed to write to the temporary urdf file with: '{}'.",
                e
            );
            panic!() // OK to panic, makes no sense to continue without a urdf.
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

    // once we have the chain, we don't need the urdf anymore
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

// actually make the kinematic chain from the urdf file (supplied or generated)
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
            panic!() // Still OK to panic, makes no sense to continue without a urdf.
        }
    }
}

// the main action service
async fn simple_robot_simulator_server(
    mut requests: impl Stream<Item = r2r::ActionServerGoalRequest<SimpleRobotControl::Action>> + Unpin,
    tf_lookup_client: r2r::Client<LookupTransform::Service>,
    chain: Chain<f64>,
    remote_control: &Arc<Mutex<bool>>,
    act_joint_state: &Arc<Mutex<JointState>>,
    ref_joint_state: &Arc<Mutex<JointState>>,
    ref_parameters: &Arc<Mutex<Parameters>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rc = *remote_control.lock().unwrap();
    loop {
        match requests.next().await {
            Some(request) => match rc {
                false => {
                    r2r::log_warn!(
                        NODE_ID,
                        "Remote Control is disabled: Simple Robot Control Action request ignored."
                    );
                    continue;
                }
                true => {
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
                            match simulate_movement(
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
                                            r2r::log_error!(
                                                NODE_ID,
                                                "Failed to abort with '{}'.",
                                                e
                                            );
                                            continue;
                                        }
                                    };
                                    continue;
                                }
                            }
                        }
                        None => {
                            r2r::log_error!(NODE_ID, "Failed to update ref values.");
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
            },
            None => (),
        }
    }
}

// offer a service to enable or dissable remote control
async fn remote_control_server(
    mut requests: impl Stream<Item = ServiceRequest<SetBool::Service>> + Unpin,
    remote_control: &Arc<Mutex<bool>>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        match requests.next().await {
            Some(request) => {
                *remote_control.lock().unwrap() = request.message.data;
                match request.message.data {
                    false => r2r::log_info!(NODE_ID, "Remote Control Disabled."),
                    true => r2r::log_info!(NODE_ID, "Remote Control Enabled."),
                }
            }
            None => (),
        }
    }
}

// offer a service to enable or dissable remote control
async fn match_ghost_server(
    mut requests: impl Stream<Item = ServiceRequest<Trigger::Service>> + Unpin,
    ref_joint_state: &Arc<Mutex<JointState>>,
    act_joint_state: &Arc<Mutex<JointState>>,
    ghost_joint_state: &Arc<Mutex<JointState>>,
    ref_parameters: &Arc<Mutex<Parameters>>
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        match requests.next().await {
            Some(_) => {
                r2r::log_info!(NODE_ID, "Got request to match ghost."); // the new ur controller should also have this
                let old_ref_js = ref_joint_state.lock().unwrap().clone();
                let mut new_ref_joint_state = old_ref_js.clone();
                new_ref_joint_state.position = ghost_joint_state.lock().unwrap().clone().position;
                r2r::log_info!(NODE_ID, "Old ref joint state: {:?}", old_ref_js);
                r2r::log_info!(NODE_ID, "Ghost joint state: {:?}", new_ref_joint_state);
                *ref_joint_state.lock().unwrap() = new_ref_joint_state;
                match simulate_movement(
                    &act_joint_state,
                    &ref_joint_state,
                    &ref_parameters,
                )
                .await
                {
                    Some(()) => (),
                    None => ()
                }
            }
            None => (),
        }
    }
}

// update the reference state based on the action request
async fn update_ref_values(
    g: ActionServerGoal<SimpleRobotControl::Action>,
    chain: &Chain<f64>,
    tf_lookup_client: &r2r::Client<LookupTransform::Service>,
    act_joint_state: &Arc<Mutex<JointState>>,
    ref_joint_state: &Arc<Mutex<JointState>>,
    ref_parameters: &Arc<Mutex<Parameters>>,
) -> Option<()> {
    // update the reference parameters
    *ref_parameters.lock().unwrap() = Parameters {
        velocity: g.goal.velocity,
        acceleration: g.goal.acceleration,
    };

    // used to transfer other parts of the joint state data
    let mut new_ref_joint_state = ref_joint_state.lock().unwrap().clone();

    match g.goal.use_joint_positions {
        // just go to the supplied joint state from the request
        true => {
            new_ref_joint_state.position = g.goal.joint_positions.position;
            *ref_joint_state.lock().unwrap() = new_ref_joint_state;
            Some(())
        }
        // now we want to go to an actual pose from the tf tree
        false => {
            // where is the goal feature frame in the robot's base frame
            let target_in_base = lookup_tf(
                &g.goal.base_frame_id,
                &g.goal.goal_feature_id,
                // g.goal.tf_lookup_deadline,
                tf_lookup_client,
            )
            .await;

            // where is the tool center point frame (or held item frame)
            // in the face plate frame of the robot
            let tcp_in_face_plate = lookup_tf(
                &g.goal.face_plate_id,
                &g.goal.tcp_id,
                // g.goal.tf_lookup_deadline,
                tf_lookup_client,
            )
            .await;

            let act_joint_state_local = &act_joint_state.lock().unwrap().clone();

            match tcp_in_face_plate {
                Some(tcp_frame) => match target_in_base {
                    Some(target_frame) => {
                        // the lookup found the transforms, let's update our chain
                        r2r::log_info!(NODE_ID, "Generating new kinematic chain.");
                        match generate_new_kinematic_chain(
                            &chain,
                            &g.goal.face_plate_id,
                            &g.goal.tcp_id,
                            &tcp_frame,
                        )
                        .await
                        {
                            Some(new_chain) => {
                                // using the new chain, find a valid joint state solution
                                r2r::log_info!(NODE_ID, "Calculating inverse kinematics.");
                                match calculate_inverse_kinematics(
                                    &new_chain,
                                    &g.goal.face_plate_id,
                                    &g.goal.tcp_id,
                                    &target_frame,
                                    &act_joint_state_local,
                                )
                                .await
                                {
                                    Some(new_joints) => {
                                        let _r = g.publish_feedback(SimpleRobotControl::Feedback {
                                            current_state: "computed inverse kinematics, moving...".into()
                                        });
                                        new_ref_joint_state.position = new_joints;
                                        *ref_joint_state.lock().unwrap() = new_ref_joint_state;
                                        Some(())
                                    }
                                    None => {
                                        r2r::log_error!(
                                            NODE_ID,
                                            "Failed to calculate inverse kinematics.",
                                        );
                                        let _r = g.publish_feedback(SimpleRobotControl::Feedback {
                                            current_state: "failed to compute inverse kinematics.".into()
                                        });
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
                            "Failed to lookup TF for: '{}' => '{}'.",
                            &g.goal.base_frame_id,
                            &g.goal.goal_feature_id,
                            // &g.goal.tf_lookup_deadline,
                        );
                        None
                    }
                },
                None => {
                    r2r::log_error!(
                        NODE_ID,
                        "Failed to lookup TF for: '{}' => '{}'.",
                        &g.goal.face_plate_id,
                        &g.goal.tcp_id,
                        // &g.goal.tf_lookup_deadline,
                    );
                    None
                }
            }
        }
    }
}

// the urdf only holds the joints and links of the robot that are always
// defined in a never-changing way. Sometimes, when the robot is expected
// to always use only one end effector and never change it, it could be reasonable
// to add a new 'fixed' joint and the end effector link to the urdf. In our
// use cases though, we would like to sometimes change tools, which changes the
// tool center point and thus the relationships to the face plate frame. Thus we
// always want to generate a new chain with the current configuration that we
// looked up from the tf. Also, an item's frame that is currently being held
// is also a reasonable tcp to be used when moving somewhere to leave the item.
async fn generate_new_kinematic_chain(
    chain: &Chain<f64>,
    face_plate_id: &str,
    tcp_id: &str,
    frame: &TransformStamped,
) -> Option<Chain<f64>> {
    // make the new face_plate to tcp joint
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
        // have to make a rot joint, a fixed one is not recognized in DoF
        .joint_type(k::JointType::Rotational {
            axis: Vector3::y_axis(),
        })
        .finalize()
        .into();

    // specify the tcp link
    let tcp_link = k::link::LinkBuilder::new().name(tcp_id).finalize();
    face_plate_to_tcp_joint.set_link(Some(tcp_link));

    // get the last joint in the chain and hope to get the right one xD
    match chain
        .iter_joints()
        .map(|j| j.name.clone())
        .collect::<Vec<String>>()
        .last()
    {
        // fetch the node that is specified by the last joint
        Some(parent) => match chain.find(parent) {
            Some(parent_node) => {
                // specify the parent of the newly made face_plate-tcp joint
                face_plate_to_tcp_joint.set_parent(parent_node);

                // get all the nodes in the chain
                let mut new_chain_nodes: Vec<k::Node<f64>> =
                    chain.iter().map(|x| x.clone()).collect();

                // add the new joint and generate the new chain
                new_chain_nodes.push(face_plate_to_tcp_joint);
                let new_chain = Chain::from_nodes(new_chain_nodes);
                Some(new_chain)
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

// get a joint position for the frame to go to
async fn calculate_inverse_kinematics(
    new_chain: &Chain<f64>,
    face_plate_id: &str,
    tcp_id: &str,
    target_frame: &TransformStamped,
    act_joint_state: &JointState,
) -> Option<Vec<f64>> {
    match new_chain.find(&format!("{}-{}", face_plate_id, tcp_id)) {
        Some(ee_joint) => {

            // a chain can have branches, but a serial chain can't
            // so we use that instead to help the solver
            let arm = k::SerialChain::from_end(ee_joint);

            // since we have added a new joint, it is now a 7DoF robot
            let mut positions = act_joint_state.position.clone(); //.lock().unwrap().clone().position;
            positions.push(0.0);

            // the solver needs an initial joint position to be set
            match arm.set_joint_positions(&positions) {
                Ok(()) => {

                    // will have to experiment with these solver parameters
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

                    // the last joint has to be rot type to be recognize, but we don't want it to rotate
                    let constraints = k::Constraints {
                        ignored_joint_names: vec![format!("{}-{}", face_plate_id, tcp_id)],
                        ..Default::default()
                    };

                    // solve, but with locking the last joint that we added
                    match solver.solve_with_constraints(&arm, &target, &constraints) {
                        Ok(()) => {

                            // get the solution and remove the seventh '0.0' joint value
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

// this task moves the act value towards the ref value
async fn simulate_movement(
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
        .map(|x| (*x * 100000.0).round() / 100000.0)
        .collect::<Vec<f64>>();

    let ref_pos = ref_joint_state
        .lock()
        .unwrap()
        .clone()
        .position
        .iter()
        .map(|x| (*x * 100000.0).round() / 100000.0)
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
                    new_act_pos.push(((y - step * z) * 100000.0).round() / 100000.0)
                } else if x > &(y + step) && z > &0.1 {
                    new_act_pos.push(((y + step * z) * 100000.0).round() / 100000.0)
                } else {
                    new_act_pos.push((*x * 100000.0).round() / 100000.0)
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

// publish the actual joint state
async fn publisher_callback(
    publisher: r2r::Publisher<JointState>,
    mut timer: r2r::Timer,
    joint_state: &Arc<Mutex<JointState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut clock = r2r::Clock::create(r2r::ClockType::RosTime).unwrap();
    loop {
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

// subscribe to the state based joint command when remote control is disabled
async fn joint_subscriber_callback(
    mut subscriber: impl Stream<Item = JointState> + Unpin,
    ref_joint_state: &Arc<Mutex<JointState>>,
    remote_control: &Arc<Mutex<bool>>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        match subscriber.next().await {
            Some(msg) => match *remote_control.lock().unwrap() {
                false => {
                    let mut new_ref_joint_state = ref_joint_state.lock().unwrap().clone();
                    new_ref_joint_state.position = msg.position;
                    *ref_joint_state.lock().unwrap() = new_ref_joint_state;
                }
                true => r2r::log_warn!(
                    NODE_ID,
                    "Remote Control is enabled: Joint State command message ignored."
                ),
            },
            None => {
                r2r::log_error!(NODE_ID, "Subscriber did not get the message?");
                ()
            }
        }
    }
}

// subscribe to the current joint pose of the ghost robot
async fn ghost_subscriber_callback(
    mut subscriber: impl Stream<Item = JointState> + Unpin,
    ghost_joint_state: &Arc<Mutex<JointState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        match subscriber.next().await {
            Some(msg) => {
                let mut new_joint_state = ghost_joint_state.lock().unwrap().clone();
                new_joint_state.position = msg.position;
                *ghost_joint_state.lock().unwrap() = new_joint_state;
            },
            None => {
                r2r::log_error!(NODE_ID, "Subscriber did not get the message?");
                ()
            }
        }
    }
}

// ask the lookup service for transforms from its buffer
async fn lookup_tf(
    parent_frame_id: &str,
    child_frame_id: &str,
    // deadline: i32,
    tf_lookup_client: &r2r::Client<LookupTransform::Service>,
) -> Option<TransformStamped> {
    let request = LookupTransform::Request {
        parent_frame_id: parent_frame_id.to_string(),
        child_frame_id: child_frame_id.to_string(),
        // deadline,
    };

    let response = tf_lookup_client
        .request(&request)
        .expect("Could not send tf Lookup request.")
        .await
        .expect("Cancelled.");

    r2r::log_info!(
        NODE_ID,
        "Request to lookup parent '{}' to child '{}' sent.",
        parent_frame_id,
        child_frame_id
    );

    match response.success {
        true => Some(response.transform),
        false => {
            r2r::log_error!(
                NODE_ID,
                "Couldn't lookup tf for parent '{}' and child '{}'.",
                parent_frame_id,
                child_frame_id
            );
            None
        }
    }
}
