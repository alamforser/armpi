#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/21
# @author:aiden
# 机械臂运动学调用
from kinematics_msgs.srv import SetRobotPose, SetJointValue

def set_pose_target(position, pitch, pitch_range=[-180.0, 180.0], resolution=1.0, duration=1.0):
    '''
    给定坐标和俯仰角，返回逆运动学解
    position: 目标位置，列表形式[x, y, z]，单位m
    pitch: 目标俯仰角，单位度，范围-180~180
    pitch_range: 如果在目标俯仰角找不到解，则在这个范围内寻找解
    resolution: pitch_range范围角度的分辨率
    return: 调用是否成功， 舵机的目标位置， 当前舵机的位置， 机械臂的目标姿态， 最优解所有舵机转动的变化量
    '''
    msg = SetRobotPose.Request()
    msg.position = [float(i) for i in position]
    msg.pitch = float(pitch)
    msg.pitch_range = [float(i) for i in pitch_range]
    msg.resolution = float(resolution)
    msg.duration = duration
    return msg

def set_joint_value_target(joint_value):
    '''
    给定每个舵机的转动角度，返回机械臂到达的目标位置姿态
    joint_value: 每个舵机转动的角度，列表形式[joint1, joint2, joint3, joint4, joint5]，单位脉宽
    return: 目标位置的3D坐标和位姿，格式geometry_msgs/Pose
    '''
    msg = SetJointValue.Request()
    msg.joint_value = [float(i) for i in joint_value]
    return msg
    
if __name__ == "__main__":
    import time
    import rclpy
    from rclpy.node import Node
    import kinematics.transform as transform
    from servo_controller.bus_servo_control import set_servo_position
    from servo_controller_msgs.msg import ServosPosition, ServoPosition
    # 初始化节点
    rclpy.init()
    node = Node('test')
    client = node.create_client(SetRobotPose, '/kinematics/set_pose_target')
    joints_pub = node.create_publisher(ServosPosition, '/servo_controller', 1)
    client.wait_for_service()
    while True:
        for i in range(10):
            msg = set_pose_target([0.18, 0.0, 0.23 + i*0.01], 0, [-90.0, 90.0], 1.0)
            future = client.call_async(msg)
            while rclpy.ok():
                rclpy.spin_once(node)
                if future.done() and future.result():
                    res = future.result()
                    break
            if res.pulse:
                servo_data = res.pulse
                set_servo_position(joints_pub, 0.1, ((1, 500), (2, 500), (3, servo_data[3]), (4, servo_data[2]), (5, servo_data[1]), (6, servo_data[0])))
                time.sleep(0.1)
        for i in range(10):
            msg = set_pose_target([0.18, 0.0, 0.33 - i*0.01], 0, [-90.0, 90.0], 1.0)
            future = client.call_async(msg)
            while rclpy.ok():
                rclpy.spin_once(node)
                if future.done() and future.result():
                    res = future.result()
                    break
            if res.pulse:
                servo_data = res.pulse
                set_servo_position(joints_pub, 0.1, ((1, 500), (2, 500), (3, servo_data[3]), (4, servo_data[2]), (5, servo_data[1]), (6, servo_data[0])))
                time.sleep(0.1)
    # while True:
        # t = time.time()
        # res = node.set_pose_target([transform.link3 + transform.tool_link, 0.0, 0.36], 0.0, [-180.0, 180.0], 1.0)
        # print(time.time() - t)
    # rclpy.logging.get_logger('p2').info(str(res[1]))
    # print('ik', res)
    # if res[1] != []:
        # res = set_joint_value_target(res[1])
        # print('fk', res)
    # node.destroy_node()
    rclpy.shutdown()
