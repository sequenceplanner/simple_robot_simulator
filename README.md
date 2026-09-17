# simple_robot_simulator

A URDF-driven kinematic robot simulator with no ROS interfaces: the robot is driven
entirely through Redis.

It loads a robot's URDF into a kinematic chain, takes goals from an external runner
over `micro_sp` state, solves inverse kinematics against the shared transform buffer,
and interpolates the joints towards the solution. The joint vector and every link
frame are published back, so the rest of the system sees a robot that moves.

The Redis interface deliberately mirrors [`ur_redis_driver`](../ur_redis_driver)'s,
key for key, so a runner can point at the simulator or at a real arm without changing
which keys it writes.

## Running

```bash
ROBOT_ID=r1 ROBOT_MODEL=ur10e DESCRIPTION_DIR=/path/to/description cargo run
```

| Env var | Default | Meaning |
|---|---|---|
| `ROBOT_ID` | `r1` | Prefix for every Redis key this simulator reads and writes |
| `ROBOT_MODEL` | `ur10e` | Picks `<DESCRIPTION_DIR>/urdf/<model>.urdf` and the mesh directory |
| `DESCRIPTION_DIR` | `description/` | URDF and meshes |
| `SIM_TF_PREFIX` | empty | Namespace for the frames this simulator publishes |
| `INITIAL_JOINT_STATE` | midpoint of each joint's limits | Comma-separated start configuration |
| `REDIS_HOST` / `REDIS_PORT` | `127.0.0.1` / `6379` | Read by `micro_sp` |

`ur_redis_driver`'s `src/ur_description/` works as a `DESCRIPTION_DIR` unchanged.

## Values in Redis

Every key is a JSON-serialised `micro_sp` `SPValue`, not a bare scalar. A string is
`{"type":"String","value":{"String":"move_j"}}`, a bool is
`{"type":"Bool","value":{"Bool":true}}`. Writing a bare `true` will fail the request
that reads it, and say which key was unreadable.

All keys below are prefixed with `ROBOT_ID`, shown here as `r1_`.

## Motion requests

Write the parameters, then set the trigger:

```
r1_command_type     the motion to run
r1_request_state    set to "initial" before triggering
r1_request_trigger  set true to submit
```

`request_state` then walks `initial → executing → succeeded | failed`, and
`r1_request_result` carries a human-readable reason. **A request always reaches a
terminal state** — a rejected request fails rather than sitting at `initial`.

Set `r1_request_cancel` true to stop the running motion. The arm holds where it is and
the request ends `succeeded` with `motion cancelled`, matching `ur_redis_driver`.

Only one goal runs at a time; a second request is rejected while one is active.

| Key | Type | Meaning |
|---|---|---|
| `r1_velocity` | float | Speed of the leading axis. `0` means `1.0` |
| `r1_use_joint_positions` / `r1_joint_positions` | bool / array | Move to joint angles directly, skipping IK and the TF lookup |
| `r1_baseframe_id` | string | Default `base_link` |
| `r1_faceplate_id` | string | Default `tool0`. The tool is mounted on **this link** |
| `r1_goal_feature_id` | string | Target frame; looked up in TF against `baseframe_id` |
| `r1_tcp_id` | string | TCP frame; looked up against `faceplate_id` |
| `r1_request_feedback` | string | Latest progress line |
| `r1_total_fail_counter` | int | Cumulative failures, never reset |
| `r1_subsequent_fail_counter` | int | Consecutive failures, reset to 0 on success |

`command_type` values that move the arm: `move_j`, `safe_move_j`, `unsafe_move_j`,
`move_l`, `safe_move_l`, `unsafe_move_l`.

Values accepted as no-ops, because they mean nothing to a kinematic simulator but a
runner rehearsing a real sequence will send them: `set_payload`, `lock_rsp`,
`unlock_rsp`, `start_vacuum`, `stop_vacuum`, `pick_vacuum`, `place_vacuum`,
`get_force`, `zero_ftsensor`. They succeed immediately without moving anything.

Anything else fails the request.

## Published state

| Key | Type | Meaning |
|---|---|---|
| `r1_joint_states` | array | Current joint positions, written when they change |
| `r1_robot_connected` | bool | True for the life of the process |
| `r1_robot_model` | string | The loaded model |

TF frames go to `tf:<child_frame_id>`: one per URDF link, plus a `<link>_visual`
frame per mesh carrying the metadata a viewer needs to draw it. These replace what
`robot_state_publisher` used to provide.

The simulator **owns** its link frames and reasserts its own parent every tick, so an
external `reparent_transform` of one of them will not survive. The root link
(`base_link`) is deliberately *not* published: where the robot stands is the scene's
business, and republishing it would overwrite the scene and drop the arm on the floor.

## Accepted and ignored

These keys exist so the interface matches `ur_redis_driver` exactly, and are read but
have no effect: `acceleration`, `global_acceleration_scaling`,
`global_velocity_scaling`, `use_execution_time`, `execution_time`, `use_blend_radius`,
`blend_radius`, `use_preferred_joint_config`, `preferred_joint_config`, `use_payload`,
`payload`, `use_relative_pose`, `relative_pose`, `force_threshold`, `root_frame_id`.

`acceleration` in particular has never been implemented — the motion profile is
constant-velocity, and the ROS version ignored it too.

The `move_l` family is **approximated in joint space**: the arm reaches the same
target, but not by a straight tool path, and it says so in `request_result`. A
rehearsal that depends on the path between waypoints will not match the real robot.

## The motion profile

Not a trapezoidal profile. Every joint's error is scaled by the largest error in the
set, so the leading axis moves a full step per 10 ms tick and the rest move a
proportional fraction — all joints start and stop together. A joint within a tenth of
the leading axis's error snaps to its target rather than crawling. Step size is
`0.1 * velocity` degrees per tick.

## Test client

```bash
cargo run --bin sim_test_client -- joints 0.0,-1.5707,1.5707,-1.5707,-1.5707,0.0
cargo run --bin sim_test_client -- frame pos1 svt_tcp
cargo run --bin sim_test_client -- lookup base_link svt_tcp
cargo run --bin sim_test_client -- capture pos1 base_link svt_tcp   # teach the current pose
cargo run --bin sim_test_client -- cancel
```

`VELOCITY` sets the speed; it defaults to `2.0`.

## What it does not do

- **No collision checking**, of the arm against itself or against the cell.
- **No joint limit enforcement.** The IK solver respects the URDF's limits, but a
  `joint_positions` request is moved to verbatim.
- **No dynamics.** `acceleration`, payload and force are all ignored; nothing here
  has mass.
- **No straight-line tool paths**, see `move_l` above.
- **The transform buffer must be acyclic.** A frame that is its own ancestor wedges
  the lookup rather than failing it, which stalls the request loop. Seed the root
  frame implicitly - name it as a parent, do not give it an entry of its own.

## Tests

```bash
cargo test
```

Covers the kinematics against `tests/fixtures/test_arm.urdf` — including that the
tool mounts on the face plate link rather than the last movable joint, which is a
distinction every UR gets wrong if you take "the last joint in the chain" — and the
interpolator's convergence, cancellation and input rejection. Nothing exercises Redis.
