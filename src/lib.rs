//! A URDF-driven kinematic robot simulator, driven entirely through Redis.
//!
//! It loads a robot's URDF into a `k::Chain`, takes goals from an external runner
//! over `micro_sp` state, solves inverse kinematics against the shared transform
//! buffer, and interpolates the joints towards the solution. The joint vector and
//! every link frame are published back, so the rest of the system sees a robot that
//! moves.
//!
//! The Redis interface deliberately mirrors `ur_redis_driver`'s, so the same runner
//! drives either one.

pub mod core;
pub mod interfaces;
pub mod kinematics;

pub use crate::core::*;
pub use interfaces::command_server::*;
pub use interfaces::state_publisher::*;
pub use kinematics::*;

/// Default frame names, matching `ur_redis_driver`.
pub static DEFAULT_BASEFRAME_ID: &str = "base_link";
pub static DEFAULT_FACEPLATE_ID: &str = "tool0";
/// Stays bare whatever the tf prefix is: there is one world, and a second station's
/// frames root into it just like the first's.
pub static DEFAULT_ROOT_FRAME_ID: &str = "world";

/// Motion commands the simulator honours by interpolating in joint space.
///
/// The `move_l` family is included even though the simulator has no straight-line
/// tool path: it reaches the same target by a different route, and says so in
/// `request_result` rather than silently pretending otherwise.
pub static MOTION_COMMANDS: &[&str] = &[
    "move_j",
    "safe_move_j",
    "unsafe_move_j",
    "move_l",
    "safe_move_l",
    "unsafe_move_l",
];

/// Commands that mean nothing to a kinematic simulator and succeed immediately.
///
/// They exist on the real driver and a runner will send them; failing them would
/// make the simulator useless for rehearsing a real sequence.
pub static NOOP_COMMANDS: &[&str] = &[
    "set_payload",
    "lock_rsp",
    "unlock_rsp",
    "start_vacuum",
    "stop_vacuum",
    "pick_vacuum",
    "place_vacuum",
    "get_force",
    "zero_ftsensor",
];

/// True if the command moves the arm in a straight tool path on the real robot.
pub fn is_linear_command(command_type: &str) -> bool {
    command_type.contains("move_l")
}
