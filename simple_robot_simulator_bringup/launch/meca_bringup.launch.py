import os
import json
import copy
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
    dir = FindPackageShare("teaching_bringup").find("teaching_bringup")

    scenario_path = {
        "scenario_path": os.path.join(dir, "scenario", "meca500"), 
    }

    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="meca_description",
            description="Description package with robot URDF/XACRO files. Usually the argument \
        is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="meca500.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value="ghost_",
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )

    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    prefix = LaunchConfiguration("prefix")


    ghost_robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(description_package), "urdf", description_file]
            ),
            " ",
            "name:=",
            "meca500",
            " ",
            "prefix:=",
            prefix,
        ]
    )

    ghost_robot_description = {"robot_description": ghost_robot_description_content}

    rviz_config_file = os.path.join(dir, "config", "bringup.rviz")

    ghost_robot_parameters = {
        "urdf_raw": ghost_robot_description_content,
        # "initial_joint_state": ["0.0", "0.0", "0.0", "0.0", "0.0", "0.0"],
        "initial_base_link_id": "ghost_" + "meca_base_link",
        "initial_face_plate_id": "ghost_" + "meca_axis_6_link",
        "initial_tcp_id": "ghost_" + "svt_tcp"
    }

    ghost_robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="ghost",
        output="screen",
        parameters=[ghost_robot_description],
        emulate_tty=True,
    )

    teaching_ghost_node = Node(
        package="teaching_ghost",
        executable="teaching_ghost",
        namespace="",
        output="screen",
        parameters=[ghost_robot_parameters],
        emulate_tty=True,
    )

    teaching_marker_node = Node(
        package="teaching_marker",
        executable="teaching_marker",
        namespace="",
        output="screen",
        parameters=[ghost_robot_parameters],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    tfbc_node = Node(
        package="teaching_tfbc",
        executable="broadcaster",
        namespace="",
        output="screen",
        parameters=[scenario_path],
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
        ghost_robot_state_publisher_node,
        teaching_ghost_node,
        rviz_node,
        tfbc_node,
        tf_lookup_node,
        teaching_marker_node
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)
