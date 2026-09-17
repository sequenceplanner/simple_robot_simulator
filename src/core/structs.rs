use micro_sp::State;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::mpsc;

/// Where the simulated robot is, where it was told to go, and what is driving it.
///
/// A plain `std::sync::Mutex`: it is never held across an `.await`.
#[derive(Default)]
pub struct SimulatorState {
    /// The joint positions the rest of the system sees, updated every sim tick.
    pub actual_joint_positions: Vec<f64>,
    /// The id of the request currently being executed, if any. Doubles as the "busy"
    /// flag that stops a second request from being admitted.
    pub goal_id: Option<String>,
    /// Signals the running motion to stop where it is.
    pub cancel_sender: Option<mpsc::Sender<()>>,
}

impl SimulatorState {
    pub fn new(initial_joint_positions: Vec<f64>) -> Self {
        Self {
            actual_joint_positions: initial_joint_positions,
            goal_id: None,
            cancel_sender: None,
        }
    }
}

/// Lock the shared state, recovering from a poisoned mutex rather than panicking.
///
/// A panic in one task should not take down a simulator that is otherwise fine: the
/// data behind this lock is a joint vector and a goal slot, both of which the next
/// writer overwrites wholesale.
pub fn lock_simulator_state(state: &Arc<Mutex<SimulatorState>>) -> MutexGuard<'_, SimulatorState> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Everything about the robot's description that the tasks need.
#[derive(Default, Clone)]
pub struct URDFParameters {
    /// The robot id, which is also the Redis key prefix.
    pub name: String,
    /// Prefix applied to every frame this simulator publishes. Empty by default.
    pub tf_prefix: String,
    pub robot_model: String,
    pub description_file: String,
    pub meshes_path: String,
}

impl URDFParameters {
    /// A frame id as it goes to Redis, i.e. with the tf prefix applied.
    pub fn frame(&self, name: &str) -> String {
        format!("{}{}", self.tf_prefix, name)
    }
}

/// One motion request, as read off Redis.
#[derive(Debug, Clone)]
pub struct RobotCommand {
    pub command_type: String,
    pub velocity: f64,
    pub acceleration: f64,
    pub use_joint_positions: bool,
    pub joint_positions: Vec<f64>,
    pub baseframe_id: String,
    pub faceplate_id: String,
    pub goal_feature_id: String,
    pub tcp_id: String,
}

// The three accessors below exist because `micro_sp`'s `State::get_value` - which
// every `get_*_or_default_*` method funnels through - logs and then **panics** when a
// key is absent, and `build_state` silently drops any key whose stored value fails to
// deserialize. Together that means one malformed value in Redis takes down the whole
// simulator instead of failing the one request that touched it. Every key an external
// process writes must be read through these.
//
// Lifted from `ur_redis_driver/src/core/structs.rs`.

pub fn state_bool_or(state: &State, key: &str, default: bool, log_target: &str) -> bool {
    if !state.contains(key) {
        log::warn!(target: log_target, "'{}' is missing or unreadable, using {}.", key, default);
        return default;
    }
    state.get_bool_or_value(key, default, log_target)
}

pub fn state_string_or(state: &State, key: &str, default: &str, log_target: &str) -> String {
    if !state.contains(key) {
        log::warn!(target: log_target, "'{}' is missing or unreadable, using '{}'.", key, default);
        return default.to_string();
    }
    state.get_string_or_value(key, default.to_string(), log_target)
}

pub fn state_int_or(state: &State, key: &str, default: i64, log_target: &str) -> i64 {
    if !state.contains(key) {
        log::warn!(target: log_target, "'{}' is missing or unreadable, using {}.", key, default);
        return default;
    }
    state.get_int_or_value(key, default, log_target)
}

pub fn state_float_or(state: &State, key: &str, default: f64, log_target: &str) -> f64 {
    if !state.contains(key) {
        log::warn!(target: log_target, "'{}' is missing or unreadable, using {}.", key, default);
        return default;
    }
    state.get_float_or_value(key, default, log_target)
}
