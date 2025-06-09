import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node  # noqa: E402
from launch import LaunchDescription, LaunchService  # noqa: E402

def generate_launch_description():
    compiled = os.environ['need_compile']
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
    camera_nodes = Node(
            package='usb_cam', 
            executable='usb_cam_node_exe', 
            output='screen',
            name='usb_cam',
            parameters=[os.path.join(peripherals_package_path, 'config', 'usb_cam_param.yaml'),
                        {"pixel_format": "yuyv2rgb"},
                        ],
            remappings = [
                ('image_raw/compressed', '/depth_cam/rgb/image_compressed'),
                ('image_raw/compressedDepth', '/depth_cam/rgb/compressedDepth'),
                ('image_raw/theora', '/depth_cam/rgb/image_raw/theora'),
                ('camera_info', '/depth_cam/rgb/camera_info'),
            ]
        )
    image_proc_node = Node(
            package='image_proc',
            executable='image_proc',
            name='image_proc_node',
            remappings=[
                        ('/image', 'image_raw'),
                        ('image_raw/compressed', '/depth_cam/rgb/image_compressed'),
                        ('image_raw/compressedDepth', '/depth_cam/rgb/compressedDepth'),
                        ('image_raw/theora', '/depth_cam/rgb/image_raw/theora'),
                        ('image_rect', '/depth_cam/rgb/image_raw'),
                        ('camera_info', '/depth_cam/rgb/camera_info'),
                        ],
            output='screen',
            arguments=['--ros-args', '--log-level', 'error']  
        )

    return LaunchDescription([camera_nodes, image_proc_node])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
