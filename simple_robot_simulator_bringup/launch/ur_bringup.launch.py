import os
import json
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    dir = FindPackageShare("simple_robot_simulator_bringup").find("simple_robot_simulator_bringup")

    robot_parameters_path = os.path.join(
        dir, "robots", "ursim10e", "general.json"
    )

    scenario_path = {
        "scenario_path": os.path.join(dir, "scenario", "ur10e"), 
    }

    with open(robot_parameters_path) as jsonfile:
        robot_parameters = json.load(jsonfile)

    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value=robot_parameters["ur_type"],
            description="Type/series of used UR robot.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="ur_description",
            description="Description package with robot URDF/XACRO files. Usually the argument \
        is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="ur.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value="",  # robot_parameters["prefix"],
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ghost_prefix",
            default_value="ghost_",  # robot_parameters["prefix"],
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )

    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    prefix = LaunchConfiguration("prefix")
    ghost_prefix = LaunchConfiguration("ghost_prefix")

    joint_limit_params = PathJoinSubstitution(
        [
            os.path.join(dir, "robots"),
            robot_parameters["name"],
            "joint_limits.yaml",
        ]
    )
    kinematics_params = PathJoinSubstitution(
        [
            os.path.join(dir, "robots"),
            robot_parameters["name"],
            "default_kinematics.yaml",
        ]
    )
    physical_params = PathJoinSubstitution(
        [
            os.path.join(dir, "robots"),
            robot_parameters["name"],
            "physical_parameters.yaml",
        ]
    )
    visual_params = PathJoinSubstitution(
        [
            os.path.join(dir, "robots"),
            robot_parameters["name"],
            "visual_parameters.yaml",
        ]
    )
    script_filename = PathJoinSubstitution(
        [FindPackageShare(description_package), "resources", "ros_control.urscript"]
    )
    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare(description_package), "resources", "rtde_input_recipe.txt"]
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare(description_package), "resources", "rtde_output_recipe.txt"]
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(description_package), "urdf", description_file]
            ),
            " ",
            "joint_limit_params:=",
            joint_limit_params,
            " ",
            "kinematics_params:=",
            kinematics_params,
            " ",
            "physical_params:=",
            physical_params,
            " ",
            "visual_params:=",
            visual_params,
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=",
            ur_type,
            " ",
            "script_filename:=",
            script_filename,
            " ",
            "input_recipe_filename:=",
            input_recipe_filename,
            " ",
            "output_recipe_filename:=",
            output_recipe_filename,
            " ",
            "prefix:=",
            prefix,
        ]
    )

    robot_description = {"robot_description": robot_description_content}

    rviz_config_file = os.path.join(dir, "config", "bringup.rviz")

    robot_parameters = {
        "urdf_raw": robot_description_content,
        "initial_joint_state": ["0.0", "-1.5707", "1.5707", "-1.5707", "-1.5707", "0.0"],
        "initial_base_link_id": "base_link",
        "initial_face_plate_id":"tool0",
        "initial_tcp_id":"svt_tcp"
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="",
        output="screen",
        parameters=[robot_description],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        emulate_tty=True,
    )

    simple_robot_simulator_node = Node(
        package="simple_robot_simulator",
        executable="simple_robot_simulator",
        namespace="",
        output="screen",
        parameters=[robot_parameters],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        emulate_tty=True,
    )

    tf_lookup_node = Node(
        package="tf_lookup",
        executable="tf_lookup",
        namespace="",
        output="screen",
        parameters=[scenario_path],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        emulate_tty=True,
    )

    tfbc_node = Node(
        package="simple_robot_simulator_tfbc",
        executable="broadcaster",
        namespace="",
        output="screen",
        parameters=[scenario_path],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        emulate_tty=True,
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        namespace="",
        output="screen",
        arguments=["-d", rviz_config_file],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        emulate_tty=True,
    )

    nodes_to_start = [
        robot_state_publisher_node,
        simple_robot_simulator_node,
        rviz_node,
        tfbc_node
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)
