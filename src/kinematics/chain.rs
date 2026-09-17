use k::nalgebra::{Translation3, UnitQuaternion, Vector3};
use k::{Chain, Node};
use crate::kinematics::convert::{isometry_to_sp_transform, sp_transform_to_isometry};
use micro_sp::{SPTransform, SPTransformStamped};
use std::error::Error;

/// A link's visual geometry, as the URDF declares it.
///
/// Carried separately from the `k::Chain` because `k` models kinematics only - it
/// drops the `<visual>` block entirely - and the visual frames this simulator
/// publishes are what give it a body in a viewer.
#[derive(Debug, Clone)]
pub struct LinkVisual {
    pub link_name: String,
    pub mesh_file: String,
    /// The visual's offset from its own link frame.
    pub origin: SPTransform,
    pub scale: f64,
}

/// A robot loaded from a URDF: the kinematic chain plus what it takes to draw it.
pub struct RobotDescription {
    pub chain: Chain<f64>,
    pub joints: Vec<String>,
    pub links: Vec<String>,
    pub visuals: Vec<LinkVisual>,
}

/// Load just the kinematic chain from a URDF.
///
/// Every caller that intends to *pose* a chain needs its own: `k::Chain` is a handle
/// over shared, interior-mutable nodes, so cloning one - or building a new `Chain`
/// from its nodes, as the TCP graft below does - hands out another view of the same
/// joint values. Two tasks calling `set_joint_positions` on chains that share nodes
/// will silently corrupt each other: an IK solve iterating against a chain that the
/// state publisher is re-posing every tick converges on nothing in particular.
pub fn load_chain(urdf_path: &str) -> Result<Chain<f64>, Box<dyn Error>> {
    Chain::<f64>::from_urdf_file(urdf_path)
        .map_err(|e| format!("failed to load the URDF at '{}': {}", urdf_path, e).into())
}

/// Load a URDF from disk.
///
/// The old ROS node took the URDF as a raw string parameter (xacro-expanded by the
/// launch file) and had to spill it into a tempfile because `k` only reads paths.
/// Nothing produces that string any more, so this reads the file directly.
pub fn load_description(urdf_path: &str) -> Result<RobotDescription, Box<dyn Error>> {
    let chain = load_chain(urdf_path)?;

    let joints = chain
        .iter_joints()
        .map(|j| j.name.clone())
        .collect::<Vec<String>>();
    let links = chain
        .iter_links()
        .map(|l| l.name.clone())
        .collect::<Vec<String>>();

    Ok(RobotDescription {
        chain,
        joints,
        links,
        visuals: read_visuals(urdf_path)?,
    })
}

/// Pull every link's visual mesh out of the URDF.
///
/// Links with no `<visual>`, or whose visual is a primitive rather than a mesh, are
/// skipped - there is nothing to draw a mesh for. A URDF that parses for `k` but not
/// for `urdf-rs` is not fatal: the arm still moves, it just has no body, so this
/// warns and returns an empty set rather than refusing to start.
fn read_visuals(urdf_path: &str) -> Result<Vec<LinkVisual>, Box<dyn Error>> {
    let robot = match urdf_rs::read_file(urdf_path) {
        Ok(r) => r,
        Err(e) => {
            log::warn!(
                target: "simple_robot_simulator",
                "Could not read visuals from '{}': {}. The robot will have no meshes.",
                urdf_path, e
            );
            return Ok(vec![]);
        }
    };

    let mut visuals = vec![];
    for link in &robot.links {
        for visual in &link.visual {
            let urdf_rs::Geometry::Mesh { filename, scale } = &visual.geometry else {
                continue;
            };

            // URDF mesh paths are usually `package://ur_description/meshes/.../base.dae`.
            // Only the file name survives here; the directory comes from the configured
            // description dir, which is what makes the same URDF work outside a ROS
            // package tree.
            let mesh_file = filename
                .rsplit('/')
                .next()
                .unwrap_or(filename.as_str())
                .to_string();

            visuals.push(LinkVisual {
                link_name: link.name.clone(),
                mesh_file,
                origin: pose_to_sp_transform(&visual.origin),
                // A non-uniform scale cannot be expressed as one factor; take x and
                // accept it, since every mesh in practice scales uniformly.
                scale: scale.map(|s| s[0]).unwrap_or(1.0),
            });
        }
    }

    Ok(visuals)
}

/// A URDF `<origin xyz rpy>` as an `SPTransform`.
fn pose_to_sp_transform(pose: &urdf_rs::Pose) -> SPTransform {
    let rotation = UnitQuaternion::from_euler_angles(pose.rpy[0], pose.rpy[1], pose.rpy[2]);
    let isometry = k::Isometry3::from_parts(
        Translation3::new(pose.xyz[0], pose.xyz[1], pose.xyz[2]),
        rotation,
    );
    isometry_to_sp_transform(isometry)
}

/// Graft a `face_plate -> tcp` joint onto the end of the chain.
///
/// The URDF models the arm up to its face plate and no further, because the tool is
/// not part of the robot: it gets swapped, and an item the gripper is holding is
/// itself a reasonable TCP to move with. So the face-plate-to-TCP relationship is
/// looked up from the transform buffer per request and the chain is rebuilt around
/// it, rather than being baked into the URDF.
pub fn generate_new_kinematic_chain(
    chain: &Chain<f64>,
    face_plate_id: &str,
    tcp_id: &str,
    frame: &SPTransformStamped,
) -> Option<Chain<f64>> {
    let isometry = sp_transform_to_isometry(&frame.transform);

    let face_plate_to_tcp_joint: Node<f64> = k::NodeBuilder::<f64>::new()
        .name(&format!("{}-{}", face_plate_id, tcp_id))
        .translation(isometry.translation)
        .rotation(isometry.rotation)
        // Has to be a rotational joint: a fixed one is not counted towards the DoF,
        // and the solver would not see it at all. It is locked again via
        // `Constraints::ignored_joint_names` when the IK actually runs.
        .joint_type(k::JointType::Rotational {
            axis: Vector3::y_axis(),
        })
        .finalize()
        .into();

    let tcp_link = k::link::LinkBuilder::new().name(tcp_id).finalize();
    face_plate_to_tcp_joint.set_link(Some(tcp_link));

    // Parent the tool to the face plate itself, found by link name.
    //
    // This used to take "the last joint in the chain" instead, which is not the same
    // node: `iter_joints` only yields movable joints, so on any robot whose face
    // plate hangs off its last axis by a *fixed* joint - which is every UR, where
    // `tool0` is fixed to `wrist_3_link` - the tool got mounted one link too early
    // and every frame goal missed by that fixed offset.
    let Some(parent_node) = chain
        .iter()
        .find(|node| node.link().as_ref().map(|l| l.name == face_plate_id) == Some(true))
        .or_else(|| {
            log::warn!(
                target: "simple_robot_simulator",
                "No link named '{}' in the URDF; mounting the tool on the last joint instead.",
                face_plate_id
            );
            chain.iter_joints().last().and_then(|j| chain.find(&j.name))
        })
    else {
        log::error!(
            target: "simple_robot_simulator",
            "Failed to find a face plate '{}' to mount the tool on.",
            face_plate_id
        );
        return None;
    };

    face_plate_to_tcp_joint.set_parent(&parent_node);

    let mut new_chain_nodes: Vec<Node<f64>> = chain.iter().map(|x| x.clone()).collect();
    new_chain_nodes.push(face_plate_to_tcp_joint);

    Some(Chain::from_nodes(new_chain_nodes))
}
