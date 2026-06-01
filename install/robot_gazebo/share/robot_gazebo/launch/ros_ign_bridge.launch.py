from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    use_sim_time = LaunchConfiguration(
        'use_sim_time',
        default='true'
    ).perform(context)

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value=use_sim_time
    )

    remappings_default = [
        ('/odom/tf', 'tf')
    ]

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Velocity command
            '/controller/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',

            # Odometry
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',

            # TF
            '/odom/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',

            # Joint states
            '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',

            # Lidar
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',

            # Registered RGB-D wrist camera
            '/depth_cam/rgbd/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/rgbd/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
        ],
        remappings=remappings_default,
        output='screen'
    )

    map_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        arguments=[
            '0.0', '0.0', '0.0',
            '0.0', '0.0', '0.0',
            'map', 'odom'
        ]
    )

    return [
        use_sim_time_arg,
        bridge,
        map_static_tf
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])


if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
