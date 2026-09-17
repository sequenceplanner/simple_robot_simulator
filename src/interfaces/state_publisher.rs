use crate::*;
// Explicit, not via the `crate::*` glob: `micro_sp::*` exports a function of the
// same name built against a different nalgebra. See `kinematics::convert`.
use crate::kinematics::convert::isometry_to_sp_transform;
use k::nalgebra::Isometry3;
use micro_sp::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Publish the simulated robot: its joint vector, and a transform per link.
///
/// This is the job `robot_state_publisher` used to do. The frames are derived from
/// the URDF rather than from a hardcoded link list, because this simulator is not
/// tied to one robot model.
pub async fn state_publisher(
    robot_params: URDFParameters,
    description: Arc<RobotDescription>,
    connection_manager: &Arc<ConnectionManager>,
    simulator_state: Arc<Mutex<SimulatorState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut con = connection_manager.get_connection().await;

    StateManager::set_sp_value(
        &mut con,
        &format!("{}_robot_model", robot_params.name),
        &robot_params.robot_model.to_spvalue(),
    )
    .await;

    // The templates carry the `parent_frame_id` and flags that the per-tick publish
    // must not lose. Keeping them here is what lets the hot path rebuild a full
    // transform without reading the old one back out of Redis first.
    let robot_transforms = initialize_robot_transforms(&description, &robot_params, &mut con).await;
    initialize_visual_transforms(&description, &robot_params, &mut con).await;

    let mut interval = runner_interval();

    // A stationary robot would otherwise rewrite the same numbers at the tick rate.
    let mut last_published: Option<Vec<f64>> = None;

    loop {
        interval.tick().await;

        let joints = lock_simulator_state(&simulator_state)
            .actual_joint_positions
            .clone();

        if last_published.as_deref() == Some(joints.as_slice()) {
            continue;
        }

        StateManager::set_sp_value(
            &mut con,
            &format!("{}_joint_states", robot_params.name),
            &SPValue::Array(ArrayOrUnknown::Array(
                joints.iter().map(|j| j.to_spvalue()).collect(),
            )),
        )
        .await;

        publish_robot_transforms(&description.chain, &joints, &robot_transforms, &mut con).await;

        last_published = Some(joints);
    }
}

/// Seed one frame per link and hand back what was written, keyed by **URDF link
/// name** rather than by the frame id.
///
/// The keying is load-bearing: `publish_robot_transforms` looks this map up with the
/// link names of the `k::Chain`, which come from the URDF and are therefore never
/// prefixed, while the values carry the prefixed ids that actually go to Redis.
/// Keying by `child_frame_id` would make every lookup miss the moment a prefix is
/// set, and since a miss is skipped rather than reported, the arm would simply never
/// move with nothing logged.
///
/// The root link is left out on purpose. Where the robot stands is the scene's
/// business - `base_link` is typically parented to a stand or a table - and
/// republishing it here would overwrite that and drop the arm on the floor.
async fn initialize_robot_transforms(
    description: &RobotDescription,
    robot_params: &URDFParameters,
    con: &mut SPConnection,
) -> HashMap<String, SPTransformStamped> {
    let mut by_urdf_name = HashMap::new();
    let mut transforms_to_insert = vec![];

    for node in description.chain.iter() {
        let Some(child) = link_name(node) else {
            continue;
        };
        let Some(parent_node) = node.parent() else {
            continue;
        };
        let Some(parent) = nearest_named_ancestor(&parent_node) else {
            continue;
        };

        let transform = SPTransformStamped {
            parent_frame_id: robot_params.frame(&parent),
            child_frame_id: robot_params.frame(&child),
            transform: SPTransform::default(),
            active_transform: true,
            enable_transform: true,
            time_stamp: SystemTime::now(),
            metadata: MapOrUnknown::UNKNOWN,
        };

        by_urdf_name.insert(child, transform.clone());
        transforms_to_insert.push(transform);
    }

    if !transforms_to_insert.is_empty() {
        let _ = TransformsManager::insert_transforms(con, &transforms_to_insert).await;
    }

    by_urdf_name
}

/// Publish a `<link>_visual` frame per mesh in the URDF.
///
/// These are static - a mesh does not move relative to the link it hangs off - so
/// they are written once at startup and never touched by the tick loop. The mesh
/// metadata is what a viewer needs to actually draw the robot.
async fn initialize_visual_transforms(
    description: &RobotDescription,
    robot_params: &URDFParameters,
    con: &mut SPConnection,
) {
    let transforms_to_insert: Vec<SPTransformStamped> = description
        .visuals
        .iter()
        .map(|visual| SPTransformStamped {
            // The prefix goes outside the suffix - `r2_shoulder_link_visual`, not
            // `shoulder_link_r2_visual`. This string is also the viewer's marker
            // namespace, so getting it wrong shows up as two robots' meshes fighting
            // over one marker.
            parent_frame_id: robot_params.frame(&visual.link_name),
            child_frame_id: robot_params.frame(&format!("{}_visual", visual.link_name)),
            transform: visual.origin.clone(),
            active_transform: false,
            enable_transform: true,
            time_stamp: SystemTime::now(),
            metadata: MapOrUnknown::Map(vec![
                (
                    "override_meshes_dir".to_spvalue(),
                    robot_params.meshes_path.to_spvalue(),
                ),
                ("mesh_file".to_spvalue(), visual.mesh_file.to_spvalue()),
                ("mesh_scale".to_spvalue(), visual.scale.to_spvalue()),
                ("visualize_mesh".to_spvalue(), true.to_spvalue()),
                ("mesh_a".to_spvalue(), 0.0.to_spvalue()),
                ("mesh_r".to_spvalue(), 0.0.to_spvalue()),
                ("mesh_g".to_spvalue(), 0.0.to_spvalue()),
                ("mesh_b".to_spvalue(), 0.0.to_spvalue()),
                ("mesh_use_embedded_materials".to_spvalue(), true.to_spvalue()),
            ]),
        })
        .collect();

    if !transforms_to_insert.is_empty() {
        let _ = TransformsManager::insert_transforms(con, &transforms_to_insert).await;
    }
}

/// Write the current kinematic frames as a single `MSET`.
///
/// One batched `insert_transforms` rather than a `move_transform` per link, which
/// would be a `GET` + deserialize + serialize + `SET` awaited one frame at a time,
/// every tick. `move_transform` only reads to preserve the parent id and metadata
/// stored in Redis; `robot_transforms` supplies those instead.
///
/// The assumption that makes explicit: this simulator owns its kinematic frames and
/// reasserts its own parent every tick, so an external `reparent_transform` of one
/// of them will not survive. The `_visual` frames are written under different child
/// ids and are never touched here.
async fn publish_robot_transforms(
    chain: &k::Chain<f64>,
    joints: &[f64],
    robot_transforms: &HashMap<String, SPTransformStamped>,
    con: &mut SPConnection,
) {
    if chain.set_joint_positions(joints).is_err() {
        return;
    }
    chain.update_transforms();

    let mut transforms_to_publish = Vec::with_capacity(robot_transforms.len());

    for node in chain.iter() {
        let Some(frame_name) = link_name(node) else {
            continue;
        };

        // Not a frame this simulator owns.
        let Some(template) = robot_transforms.get(&frame_name) else {
            continue;
        };

        let child_world = node.world_transform().unwrap_or_else(Isometry3::identity);

        // Frames are stored relative to their parent, so undo the parent's pose.
        // Walking to the nearest *named* ancestor rather than the immediate one keeps
        // this consistent with how the parent ids were assigned at startup.
        let relative = match nearest_named_ancestor_node(node) {
            Some(parent) => {
                let parent_world = parent.world_transform().unwrap_or_else(Isometry3::identity);
                parent_world.inverse() * child_world
            }
            None => child_world,
        };

        let mut updated = template.clone();
        updated.transform = isometry_to_sp_transform(relative);
        updated.time_stamp = SystemTime::now();
        transforms_to_publish.push(updated);
    }

    // `insert_transforms` logs an error on an empty vector, and this runs at the
    // tick rate, so do not hand it one.
    if !transforms_to_publish.is_empty() {
        let _ = TransformsManager::insert_transforms(con, &transforms_to_publish).await;
    }
}

/// The URDF link a chain node carries, if it carries one.
///
/// `k` models a chain as joint nodes, and only some of them have a link attached.
/// A node without one is a kinematic intermediate with no frame to publish.
fn link_name(node: &k::Node<f64>) -> Option<String> {
    node.link().as_ref().map(|link| link.name.clone())
}

/// Walk up until a node that carries a link is found.
fn nearest_named_ancestor_node(node: &k::Node<f64>) -> Option<k::Node<f64>> {
    let mut current = node.parent();
    while let Some(candidate) = current {
        if link_name(&candidate).is_some() {
            return Some(candidate);
        }
        current = candidate.parent();
    }
    None
}

fn nearest_named_ancestor(node: &k::Node<f64>) -> Option<String> {
    if let Some(name) = link_name(node) {
        return Some(name);
    }
    nearest_named_ancestor_node(node).and_then(|n| link_name(&n))
}
