#!/usr/bin/env python3
# coding: utf8
# 手眼标定程序，适配特定 set_servo_position 调用格式

import os
import cv2
import yaml
import time
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from kinematics_msgs.srv import GetRobotPose, SetJointValue
from std_srvs.srv import Trigger
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from dt_apriltags import Detector
import threading
from sdk import common  # 假设存在 common 模块，包含 xyz_quat_to_mat 等函数
from servo_controller.bus_servo_control import set_servo_position
from kinematics.kinematics_control import set_joint_value_target

class HandEyeCalibration(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("初始化手眼标定节点")
        
        # 初始化变量
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.intrinsic = None  # 相机内参
        self.distortion = None  # 畸变系数
        self.tag_size = 0.025  # AprilTag 尺寸（米）
        self.R_gripper2base = []  # 机器人末端到基座的旋转矩阵列表
        self.t_gripper2base = []  # 机器人末端到基座的平移向量列表
        self.R_target2cam = []  # 标定目标到相机的旋转矩阵列表
        self.t_target2cam = []  # 标定目标到相机的平移向量列表
        self.calibration_poses = []  # 预定义的标定位姿（关节角度）
        self.current_pose_index = 0
        self.calibrating = False
        
        # AprilTag 检测器
        self.at_detector = Detector(
            searchpath=['apriltags'],
            families='tag36h11',
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # ROS2 话题和服务
        self.image_sub = self.create_subscription(
            Image, '/depth_cam/rgb/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/depth_cam/rgb/camera_info', self.info_callback, 10)
        self.pose_client = self.create_client(
            GetRobotPose, '/kinematics/get_current_pose')
        self.set_joint_client = self.create_client(
            SetJointValue, '/kinematics/set_joint_value_target')
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        
        # 等待服务
        self.get_logger().info("等待运动学服务...")
        if not self.pose_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("服务 /kinematics/get_current_pose 不可用")
            raise RuntimeError("运动学服务不可用")
        if not self.set_joint_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("服务 /kinematics/set_joint_value_target 不可用")
            raise RuntimeError("运动学服务不可用")

        # 标定服务
        self.calibration_srv = self.create_service(
            Trigger, '~/start_calibration', self.start_calibration_callback)
        
        # 测试移动服务
        self.test_move_srv = self.create_service(
            Trigger, '~/test_move', self.test_move_callback)
        
        # 配置路径
        self.config_path = "/home/ubuntu/ros2_ws/src/app/config/"
        self.config_file = "transform.yaml"

        # 定义多个关节角度位姿，适配 set_servo_position (伺服 ID: 6,5,4,3,2,1)
        self.calibration_poses = [
            [500, 600, 785, 110, 500, 210],  # 初始位姿，调整 jointangles[2] 为 785
            [500, 650, 750, 120, 500, 210],  # 自定义位姿 1
            [500, 550, 785, 100, 500, 210],  # 自定义位姿 2
            [500, 600, 785, 110, 500, 210],  # 自定义位姿 3
            [500, 700, 750, 130, 500, 210],  # 自定义位姿 4
            [500, 500, 785, 100, 500, 210],  # 自定义位姿 5
            [500, 600, 785, 110, 500, 210],  # 自定义位姿 6
            [500, 650, 750, 120, 500, 210],  # 自定义位姿 7
            [500, 550, 785, 100, 500, 210],  # 自定义位姿 8
            [500, 600, 750, 110, 500, 210]   # 自定义位姿 9
        ]

        # 初始化 OpenCV 窗口
        cv2.namedWindow("AprilTag Detection", cv2.WINDOW_NORMAL)

    def info_callback(self, msg):
        """处理相机内参和畸变系数"""
        with self.lock:
            self.intrinsic = np.array(msg.k).reshape(3, 3)
            self.distortion = np.array(msg.d)
            self.get_logger().info("收到相机内参")
            self.get_logger().info(f"内参矩阵:\n{self.intrinsic}")
            self.get_logger().info(f"畸变系数: {self.distortion}")

    def image_callback(self, msg):
        """处理相机图像并进行 AprilTag 检测，显示检测结果"""
        if not self.calibrating:
            self.get_logger().debug("未启动标定，跳过图像处理")
            return
        if self.intrinsic is None or self.distortion is None:
            self.get_logger().warn("缺少相机内参，跳过图像处理")
            return
        
        # 将 ROS 图像转换为 OpenCV 格式
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return
        
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # 检测 AprilTag
        try:
            tags = self.at_detector.detect(
                gray, True, 
                (self.intrinsic[0, 0], self.intrinsic[1, 1], 
                 self.intrinsic[0, 2], self.intrinsic[1, 2]), 
                self.tag_size
            )
        except Exception as e:
            self.get_logger().error(f"AprilTag 检测失败: {e}")
            return
        
        # 可视化 AprilTag
        for tag in tags:
            corners = tag.corners.astype(int)
            cv2.polylines(bgr_image, [corners], True, (0, 255, 0), 2)
            cv2.putText(bgr_image, f"ID: {tag.tag_id}", tuple(corners[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 显示图像
        cv2.imshow("AprilTag Detection", bgr_image)
        cv2.waitKey(1)
        
        # 保存图像以便调试
        cv2.imwrite(f"/tmp/apriltag_{self.current_pose_index}.png", bgr_image)
        self.get_logger().info(f"已保存图像: /tmp/apriltag_{self.current_pose_index}.png")
        
        if not tags:
            self.get_logger().warn("未检测到 AprilTag")
            return
        
        # 获取机器人当前位姿
        try:
            pose = self.send_request(self.pose_client, GetRobotPose.Request())
            pose_t = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            pose_r = [pose.pose.orientation.w, pose.pose.orientation.x, 
                      pose.pose.orientation.y, pose.pose.orientation.z]
            self.get_logger().info(f"当前机器人位姿: 位置={pose_t}, 姿态={pose_r}")
        except Exception as e:
            self.get_logger().error(f"获取机器人位姿失败: {e}")
            return
        
        # 转换为旋转矩阵
        try:
            R_gripper2base = common.xyz_quat_to_mat(pose_t, pose_r)[:3, :3]
            t_gripper2base = np.array(pose_t)
        except Exception as e:
            self.get_logger().error(f"位姿转换失败: {e}")
            return
        
        # 保存位姿数据
        with self.lock:
            self.R_gripper2base.append(R_gripper2base)
            self.t_gripper2base.append(t_gripper2base)
            self.R_target2cam.append(tags[0].pose_R)
            self.t_target2cam.append(tags[0].pose_t.flatten())
            self.get_logger().info(f"已收集 {len(self.R_gripper2base)} 组位姿数据")
        
        # 移动到下一个位姿
        self.current_pose_index += 1
        if self.current_pose_index < len(self.calibration_poses):
            self.move_to_pose(self.calibration_poses[self.current_pose_index])
        else:
            # 完成标定
            self.calibrate()
            self.calibrating = False
            cv2.destroyAllWindows()

    def send_request(self, client, msg):
        """发送服务请求并等待结果"""
        try:
            future = client.call_async(msg)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is None:
                raise RuntimeError("服务调用失败")
            return future.result()
        except Exception as e:
            self.get_logger().error(f"服务请求失败: {e}")
            raise

    def move_to_pose(self, joint_angles):
        """移动机器人到指定关节角度位姿，适配 set_servo_position"""
        with self.lock:
            self.get_logger().info(f"尝试移动到关节角度: {joint_angles}")
            try:
                # 调用运动学服务
                msg = set_joint_value_target(joint_angles)
                response = self.send_request(self.set_joint_client, msg)
                self.get_logger().info(f"服务 /kinematics/set_joint_value_target 调用成功: {response}")
                
                # 设置伺服位置，适配提供的格式
                servo_positions = (
                    (6, joint_angles[0]),  # joint_angle[0]
                    (5, joint_angles[1]),  # joint_angle[1]
                    (4, joint_angles[2]),  # joint_angle[2]
                    (3, joint_angles[3]),  # joint_angle[3]
                    (2, joint_angles[4]),  # joint_angle[4]
                    (1, joint_angles[5])   # joint_angle[5]
                )
                set_servo_position(self.joints_pub, 1.5, servo_positions)
                self.get_logger().info(f"已发布伺服指令: {servo_positions}")
                time.sleep(1.5)  # 确保机械臂移动到位
            except Exception as e:
                self.get_logger().error(f"移动机器人失败: {e}")

    def start_calibration_callback(self, request, response):
        """触发标定服务"""
        with self.lock:
            if not self.calibrating:
                self.get_logger().info("开始手眼标定")
                self.calibrating = True
                self.R_gripper2base, self.t_gripper2base = [], []
                self.R_target2cam, self.t_target2cam = [], []
                self.current_pose_index = 0
                self.move_to_pose(self.calibration_poses[0])
            else:
                self.get_logger().warn("标定已在进行中")
        
        response.success = True
        response.message = "标定开始"
        return response

    def test_move_callback(self, request, response):
        """测试机械臂移动到第一个标定位姿"""
        self.get_logger().info("触发测试移动")
        try:
            self.move_to_pose(self.calibration_poses[0])
            response.success = True
            response.message = "测试移动成功"
        except Exception as e:
            self.get_logger().error(f"测试移动失败: {e}")
            response.success = False
            response.message = f"测试移动失败: {str(e)}"
        return response

    def calibrate(self):
        """执行手眼标定并保存结果"""
        with self.lock:
            if len(self.R_gripper2base) < 3:
                self.get_logger().error("位姿数据不足，无法标定")
                return
            
            # 执行手眼标定
            self.get_logger().info("执行手眼标定计算")
            try:
                R_hand2cam, t_hand2cam = cv2.calibrateHandEye(
                    self.R_gripper2base, self.t_gripper2base,
                    self.R_target2cam, self.t_target2cam,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )
                
                # 计算重投影误差
                reproj_error = 0.0
                for i in range(len(self.R_gripper2base)):
                    H_gripper2base = np.eye(4)
                    H_gripper2base[:3, :3] = self.R_gripper2base[i]
                    H_gripper2base[:3, 3] = self.t_gripper2base[i]
                    H_target2cam = np.eye(4)
                    H_target2cam[:3, :3] = self.R_target2cam[i]
                    H_target2cam[:3, 3] = self.t_target2cam[i]
                    H_hand2cam = np.eye(4)
                    H_hand2cam[:3, :3] = R_hand2cam
                    H_hand2cam[:3, 3] = t_hand2cam.flatten()
                    H_error = np.dot(np.dot(H_gripper2base, H_hand2cam), np.linalg.inv(H_target2cam))
                    error = np.linalg.norm(H_error[:3, 3])
                    reproj_error += error
                reproj_error /= len(self.R_gripper2base)
                self.get_logger().info(f"平均重投影误差: {reproj_error:.6f} 米")
                
                # 保存结果
                hand2cam_tf_matrix = np.eye(4)
                hand2cam_tf_matrix[:3, :3] = R_hand2cam
                hand2cam_tf_matrix[:3, 3] = t_hand2cam.flatten()
                self.get_logger().info(f"标定结果 hand2cam_tf_matrix:\n{hand2cam_tf_matrix}")
                
                config_path = os.path.join(self.config_path, self.config_file)
                config = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                config['hand2cam_tf_matrix'] = hand2cam_tf_matrix.tolist()
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config, f)
                self.get_logger().info(f"已保存标定结果到 {config_path}")
            except Exception as e:
                self.get_logger().error(f"手眼标定计算失败: {e}")

def main():
    rclpy.init()
    try:
        node = HandEyeCalibration('hand_eye_calibration')
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
