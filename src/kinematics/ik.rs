use k::prelude::InverseKinematicsSolver;
use k::{Chain, SerialChain};
use crate::kinematics::convert::sp_transform_to_isometry;
use micro_sp::SPTransformStamped;

/// Solve for a joint configuration that puts `tcp_id` on `target_frame`.
///
/// `new_chain` must be the chain returned by
/// [`crate::kinematics::generate_new_kinematic_chain`] - the solve is done against
/// the synthetic `face_plate-tcp` joint, which is what makes the target a TCP pose
/// rather than a face-plate pose.
pub fn calculate_inverse_kinematics(
    new_chain: &Chain<f64>,
    face_plate_id: &str,
    tcp_id: &str,
    target_frame: &SPTransformStamped,
    seed_joint_positions: &[f64],
    log_target: &str,
) -> Option<Vec<f64>> {
    let ee_joint_name = format!("{}-{}", face_plate_id, tcp_id);
    let Some(ee_joint) = new_chain.find(&ee_joint_name) else {
        log::error!(target: log_target, "Failed to find the end effector joint '{}'.", ee_joint_name);
        return None;
    };

    // A chain may branch; a serial chain cannot. Handing the solver the serial form
    // is what keeps it from wandering into a branch that does not reach the tool.
    let arm = SerialChain::from_end(ee_joint);

    // The grafted joint makes this a 7-DoF arm as far as the solver is concerned, so
    // the seed needs a seventh value. It is popped off again below.
    let mut positions = seed_joint_positions.to_vec();
    positions.push(0.0);

    if let Err(e) = arm.set_joint_positions(&positions) {
        log::error!(target: log_target, "Failed to seed the solver with the current joint positions: {}.", e);
        return None;
    }

    let solver = k::JacobianIkSolver::new(0.01, 0.01, 0.5, 50);
    let target = sp_transform_to_isometry(&target_frame.transform);

    // The grafted joint has to be rotational to be counted in the DoF, but it must
    // not actually rotate - it represents a rigid tool mount.
    let constraints = k::Constraints {
        ignored_joint_names: vec![ee_joint_name],
        ..Default::default()
    };

    if let Err(e) = solver.solve_with_constraints(&arm, &target, &constraints) {
        log::error!(target: log_target, "Failed to solve the inverse kinematics: {}.", e);
        return None;
    }

    let mut solution = arm.joint_positions();
    match solution.pop() {
        Some(_) => Some(solution),
        None => {
            log::error!(target: log_target, "Failed to shrink the solution back to the robot's DoF.");
            None
        }
    }
}
