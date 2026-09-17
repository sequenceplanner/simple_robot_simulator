//! Kinematics against the bundled test URDF.

use micro_sp::{MapOrUnknown, SPRotation, SPTransform, SPTransformStamped, SPTranslation};
use ordered_float::OrderedFloat;
use simple_robot_simulator::*;
use simple_robot_simulator::kinematics::convert::isometry_to_sp_transform;
use std::time::SystemTime;

const URDF: &str = "tests/fixtures/test_arm.urdf";

fn transform(x: f64, y: f64, z: f64) -> SPTransformStamped {
    SPTransformStamped {
        active_transform: true,
        enable_transform: true,
        time_stamp: SystemTime::now(),
        parent_frame_id: "base_link".to_string(),
        child_frame_id: "target".to_string(),
        transform: SPTransform {
            translation: SPTranslation {
                x: OrderedFloat(x),
                y: OrderedFloat(y),
                z: OrderedFloat(z),
            },
            rotation: SPRotation {
                x: OrderedFloat(0.0),
                y: OrderedFloat(0.0),
                z: OrderedFloat(0.0),
                w: OrderedFloat(1.0),
            },
        },
        metadata: MapOrUnknown::UNKNOWN,
    }
}

#[test]
fn loads_joints_links_and_visuals_from_the_urdf() {
    let description = load_description(URDF).expect("the fixture URDF should load");

    assert_eq!(
        description.joints,
        vec![
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "only the movable joints are part of the DoF"
    );

    assert!(description.links.contains(&"tool0".to_string()));

    // Three links in the fixture carry a mesh; the rest have no <visual> at all.
    let meshes: Vec<&str> = description
        .visuals
        .iter()
        .map(|v| v.mesh_file.as_str())
        .collect();
    assert_eq!(
        meshes,
        vec!["shoulder.dae", "upperarm.dae", "forearm.dae"],
        "the package:// path should be reduced to a bare file name"
    );
}

#[test]
fn visual_origin_is_read_from_the_urdf() {
    let description = load_description(URDF).unwrap();
    let upper_arm = description
        .visuals
        .iter()
        .find(|v| v.link_name == "upper_arm_link")
        .expect("upper_arm_link has a visual");

    assert!((upper_arm.origin.translation.z.into_inner() - 0.1).abs() < 1e-9);
    // rpy="1.5707 0 0" is a quarter turn about x, so w and x should both be ~sqrt(2)/2.
    assert!((upper_arm.origin.rotation.x.into_inner() - 0.70710).abs() < 1e-4);
}

#[test]
fn grafting_the_tcp_joint_extends_the_chain() {
    let description = load_description(URDF).unwrap();
    let before = description.chain.iter_joints().count();

    let new_chain = generate_new_kinematic_chain(
        &description.chain,
        "tool0",
        "svt_tcp",
        &transform(0.0, 0.0, 0.15),
    )
    .expect("the tcp joint should graft onto the chain");

    assert_eq!(
        new_chain.iter_joints().count(),
        before + 1,
        "the synthetic joint has to be counted in the DoF or the solver ignores it"
    );
    assert!(new_chain.find("tool0-svt_tcp").is_some());
}

#[test]
fn the_tool_is_mounted_on_the_face_plate_not_the_last_axis() {
    let description = load_description(URDF).unwrap();

    // `tool0` hangs off `wrist_3_link` by a *fixed* joint, so it is not in
    // `iter_joints()`. Mounting the tool on the last movable joint instead - which is
    // what this used to do - silently drops that fixed offset and every frame goal
    // misses by it.
    let new_chain = generate_new_kinematic_chain(
        &description.chain,
        "tool0",
        "svt_tcp",
        &transform(0.0, 0.0, 0.12),
    )
    .unwrap();

    let tcp = new_chain.find("tool0-svt_tcp").unwrap();
    let parent = tcp.parent().expect("the tool must have a parent");
    let parent_link = parent.link().as_ref().map(|l| l.name.clone());

    assert_eq!(
        parent_link,
        Some("tool0".to_string()),
        "the tool has to hang off the face plate link named in the request"
    );
}

/// Pose the arm, read where the tool ends up, then ask the solver to get back there.
///
/// Picking a target by hand tends to pick one that is unreachable in *orientation*
/// even when the position is well inside the workspace, which tests the solver's
/// failure path by accident rather than its success path.
#[test]
fn inverse_kinematics_recovers_a_pose_reached_by_forward_kinematics() {
    let description = load_description(URDF).unwrap();
    let tcp_offset = transform(0.0, 0.0, 0.12);

    let posed = generate_new_kinematic_chain(&description.chain, "tool0", "svt_tcp", &tcp_offset)
        .unwrap();

    let truth = vec![0.3, -1.1, 1.2, -1.6, -1.5, 0.2];
    let mut with_synthetic = truth.clone();
    with_synthetic.push(0.0);
    posed.set_joint_positions(&with_synthetic).unwrap();
    posed.update_transforms();

    let target_isometry = posed
        .find("tool0-svt_tcp")
        .unwrap()
        .world_transform()
        .unwrap();

    let mut target = transform(0.0, 0.0, 0.0);
    target.transform = isometry_to_sp_transform(target_isometry);

    // A fresh chain, so the solve starts from the seed rather than from the answer.
    let solve_chain = load_chain(URDF).unwrap();
    let new_chain =
        generate_new_kinematic_chain(&solve_chain, "tool0", "svt_tcp", &tcp_offset).unwrap();

    let seed = vec![0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0];
    let solution =
        calculate_inverse_kinematics(&new_chain, "tool0", "svt_tcp", &target, &seed, "test")
            .expect("a pose the arm demonstrably reaches must solve");

    assert_eq!(solution.len(), seed.len());

    // Check the pose, not the joint values: a 6-DoF arm has several ways to hold the
    // same tool pose and any of them is a correct answer.
    let check = load_chain(URDF).unwrap();
    let check_chain =
        generate_new_kinematic_chain(&check, "tool0", "svt_tcp", &tcp_offset).unwrap();
    let mut solved = solution.clone();
    solved.push(0.0);
    check_chain.set_joint_positions(&solved).unwrap();
    check_chain.update_transforms();
    let reached = check_chain
        .find("tool0-svt_tcp")
        .unwrap()
        .world_transform()
        .unwrap();

    let error = (reached.translation.vector - target_isometry.translation.vector).norm();
    assert!(
        error < 0.01,
        "the solved configuration should put the tool within a centimetre of the target, got {}",
        error
    );
}

#[test]
fn inverse_kinematics_fails_for_an_unreachable_pose() {
    let description = load_description(URDF).unwrap();
    let seed = vec![0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0];

    let new_chain = generate_new_kinematic_chain(
        &description.chain,
        "tool0",
        "svt_tcp",
        &transform(0.0, 0.0, 0.1),
    )
    .unwrap();

    // Ten metres away: far outside a roughly one-metre arm.
    let solution = calculate_inverse_kinematics(
        &new_chain,
        "tool0",
        "svt_tcp",
        &transform(10.0, 10.0, 10.0),
        &seed,
        "test",
    );

    assert!(
        solution.is_none(),
        "an unreachable pose must fail the request rather than return a bad solution"
    );
}
