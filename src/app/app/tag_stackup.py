#!/usr/bin/env python3
# coding: utf8
#标签码垛

import os
import cv2
import copy
import yaml
import time
import math
import queue
import rclpy
import threading
import numpy as np
from sdk import common, fps
from rclpy.node import Node
from app.common import Heart
from cv_bridge import CvBridge
from dt_apriltags import Detector
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from interfaces.srv import SetStringBool
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from kinematics.kinematics_control import set_joint_value_target
from servo_controller.bus_servo_control import set_servo_position
from app.utils import calculate_grasp_yaw, position_change_detect, pick_and_place, image_process, distortion_inverse_map



class TagStackup(Node):
    hand2cam_tf_matrix = [
    [0.0, 0.0, 1.0, -0.101],
    [-1.0, 0.0, 0.0, 0.01],
    [0.0, -1.0, 0.0, 0.05],
    [0.0, 0.0, 0.0, 1.0]
]
    place_position = [-0.006, 0.16, 0.015]

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.tag_size = 0.025
        self.target_miss_count = 0
        self.count_move = 0
        self.count_still = 0
        self.running = True
        self.enter = False
        self.get_height = False
        self.intrinsic = None
        self.last_object_info_list = None
        self.endpoint = None
        self.start_get_roi = False
        self.transport_info = None
        self.start_transport = False
        self.distortion = None
        self.extristric = None
        self.white_area_center = None
        self.roi = None
        self.enable_stackup = False
        self.last_position = None
        self.target = None
        self.count = 0
        self.err_msg = None
        self.target_labels = ["tag1", "tag2", "tag3"]

        self.camera_type = os.environ['CAMERA_TYPE']
        self.config_file = 'transform.yaml'
        self.calibration_file = 'calibration.yaml'
        self.config_path = "/home/ubuntu/ros2_ws/src/app/config/"
        self.image_queue = queue.Queue(maxsize=2)

        self.lock = threading.RLock()
        self.fps = fps.FPS()  # 帧率统计器(frame rate counter)
        self.bridge = CvBridge()  # 用于ROS Image消息与OpenCV图像之间的转换

        self.at_detector = Detector(searchpath=['apriltags'], 
                                    families='tag36h11',
                                    nthreads=4,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        # services and topics
        self.image_sub = None
        self.camera_info_sub = None

        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result',  1)
        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_stack_up_srv = self.create_service(SetBool, '~/enable_stackup', self.enable_stackup_srv_callback)

        timer_cb_group = ReentrantCallbackGroup()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()


        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)
    
    def get_node_state(self, request, response):
        response.success = True
        return response

    def init_process(self):
        self.timer.cancel()

        threading.Thread(target=self.main, daemon=True).start()
        threading.Thread(target=self.transport_thread(), daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_endpoint(self):

        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        self.get_logger().info('get endpoint: %s' % endpoint)
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation

        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "Loading tag stackup")
        with self.lock:
            self.enter = True
            self.heart = Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(None, response=Trigger.Response()))  # 心跳包(heartbeat package)
            self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
            self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1)
            joint_angle = [500, 600, 800, 110, 500, 210]
        
            set_servo_position(self.joints_pub, 1, ((5, joint_angle[1]), (4, joint_angle[2]), (3, joint_angle[3]), (2, joint_angle[4]), (1, joint_angle[5])))
            time.sleep(1)

            set_servo_position(self.joints_pub, 1, ((6, joint_angle[0]), ))
            time.sleep(1)
        self.start_get_roi = True

        response.success = True
        response.message = "enter"
        return response

    def camera_info_callback(self, msg):
        self.intrinsic = np.matrix(msg.k).reshape(1, -1, 3)
        self.distortion = np.array(msg.d)

    def exit_srv_callback(self, request, response):
        if self.enter and request is not None:
            self.get_logger().info('\033[1;32m%s\033[0m' % "exit tag stackup")
            with self.lock:
                self.enter = False
                self.start_transport = False
                self.enable_stackup = False
                try:
                    if self.image_sub is not None:
                        self.destroy_subscription(self.image_sub)
                        self.image_sub = None
                    if self.camera_info_sub is not None:
                        self.destroy_subscription(self.camera_info_sub)
                        self.camera_info_sub = None
                except Exception as e:
                    self.get_logger().error(str(e))
                self.heart.destroy()
                self.heart = None
                self.err_msg = None
                pick_and_place.interrupt(True)
        elif not self.enter and request is None:
            self.get_logger().info('\033[1;32m%s\033[0m' % "heart already stop")

        response.success = True
        response.message = "exit"
        return response

    def enable_stackup_srv_callback(self, request, response):
        with self.lock:
            if request.data:
                self.get_logger().info('\033[1;32m%s\033[0m' % "start tag stackup")             
                self.enable_stackup = True
                self.last_position = None
                self.get_endpoint()
                self.go_left()
                pick_and_place.interrupt(False)

            else:
                self.get_logger().info('\033[1;32m%s\033[0m' % "stop tag stackup")
                self.enable_stackup = False
                pick_and_place.interrupt(True)
                self.err_msg = None
        response.success = True
        response.message = "start"
        return response

    def go_home(self):
        #joint_angle = [500, 600, 800, 110, 500, 210]
        joint_angle = [500, 600, 800, 110, 500, 210]
    
        set_servo_position(self.joints_pub, 1, ((5, joint_angle[1]), (4, joint_angle[2]), (3, joint_angle[3]), (2, joint_angle[4]), (1, joint_angle[5])))
        time.sleep(1)

        set_servo_position(self.joints_pub, 1, ((6, joint_angle[0]), ))
        time.sleep(1)

    def go_left(self):
        joint_angle = [875, 600, 800, 110, 500, 210]
    
        set_servo_position(self.joints_pub, 1, ((6, joint_angle[0]), (5, joint_angle[1]), (4, joint_angle[2]), (3, joint_angle[3]), (2, joint_angle[4]), (1, joint_angle[5])))
        time.sleep(1)
        self.get_height = True

    def image_callback(self, ros_image):
        # 将ros格式图像转换为opencv格式(convert the ros format image to opencv format)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)

        if self.image_queue.full():
            # # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
        # # 将图像放入队列
        self.image_queue.put(bgr_image)

    def get_roi(self):
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 转换为 numpy 数组
            extristric = np.array(config['extristric'])
            corners = np.array(config['corners']).reshape(-1, 3)
            self.white_area_center = np.array(config['white_area_pose_world'])
        while True:
            intrinsic = self.intrinsic
            distortion = self.distortion
            if intrinsic is not None and distortion is not None:
                break
            time.sleep(0.1)

        tvec = extristric[:1]  # 取第一行
        rmat = extristric[1:]  # 取后面三行

        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.03)
        self.extristric = tvec, rmat
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.04)
        imgpts, jac = cv2.projectPoints(corners[:-1], np.array(rmat), np.array(tvec), intrinsic, distortion)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # 裁切出ROI区域(crop RIO region)
        x_min = min(imgpts, key=lambda p: p[0])[0] # x轴最小值(the minimum value of X-axis)
        x_max = max(imgpts, key=lambda p: p[0])[0] # x轴最大值(the maximum value of X-axis)
        y_min = min(imgpts, key=lambda p: p[1])[1] # y轴最小值(the minimum value of Y-axis)
        y_max = max(imgpts, key=lambda p: p[1])[1] # y轴最大值(the maximum value of Y-axis)
        roi = np.maximum(np.array([y_min, y_max, x_min, x_max]), 0)
            
        self.roi = roi

    def get_object_world_position(self, position, intrinsic, extristric, white_area_center, height=0.03):
        projection_matrix = np.row_stack((np.column_stack((extristric[1], extristric[0])), np.array([[0, 0, 0, 1]])))
        world_pose = common.pixels_to_world([position], intrinsic, projection_matrix)[0]
        world_pose[0] = -world_pose[0]
        world_pose[1] = -world_pose[1]
        position = white_area_center[:3, 3] + world_pose
        position[2] = height
        
        config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
        offset = tuple(config_data['pixel']['offset'])
        scale = tuple(config_data['pixel']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i]
            position[i] = position[i] + offset[i]
        return position, projection_matrix

    def calculate_pick_grasp_yaw(self, position, target, target_info, intrinsic, projection_matrix):
        yaw = math.degrees(math.atan2(position[1], position[0]))
        if position[0] < 0 and position[1] < 0:
            yaw = yaw + 180
        elif position[0] < 0 and position[1] > 0:
            yaw = yaw - 180
        # 0.09x0.02
        gripper_size = [common.calculate_pixel_length(0.09, intrinsic, projection_matrix),
                        common.calculate_pixel_length(0.02, intrinsic, projection_matrix)]

        return calculate_grasp_yaw.calculate_gripper_yaw_angle(target, target_info, gripper_size, yaw)

    def calculate_place_grasp_yaw(self, position, angle=0):
        yaw = math.degrees(math.atan2(position[1], position[0]))
        if position[0] < 0 and position[1] < 0:
            yaw = yaw + 180
        elif position[0] < 0 and position[1] > 0:
            yaw = yaw - 180
        yaw1 = yaw + angle
        if yaw < 0:
            yaw2 = yaw1 + 90
        else:
            yaw2 = yaw1 - 90

        yaw = yaw2
        if abs(yaw1) < abs(yaw2):
            yaw = yaw1
        yaw = 500 + int(yaw / 240 * 1000)

        return yaw

    def transport_thread(self):
        while self.running:
            if self.start_transport:
                position, yaw, target = self.transport_info
                if position[0] > 0.22:
                    position[2] += 0.01
                config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                offset = tuple(config_data['kinematics']['offset'])
                scale = tuple(config_data['kinematics']['scale'])
                for i in range(3):
                    position[i] = position[i] * scale[i]
                    position[i] = position[i] + offset[i]
                # self.get_logger().info(f'pick2:{position}')

                finish = pick_and_place.pick(position, 80, yaw, 540, 0.02, self.joints_pub, self.kinematics_client)
                if finish:
                    position = copy.deepcopy(self.place_position)

                    yaw = self.calculate_place_grasp_yaw(position, 0)
                    config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                    offset = tuple(config_data['kinematics']['offset'])
                    scale = tuple(config_data['kinematics']['scale'])
                    angle = math.degrees(math.atan2(position[1], position[0]))
                    if angle > 45:
                        # self.get_logger().info(f'1:{position}')
                        position = [position[0] * scale[1], position[1] * scale[0], position[2] * scale[2]]
                        position = [position[0] - offset[1], position[1] + offset[0], position[2] + offset[2]]
                    elif angle < -45:
                        # self.get_logger().info(f'2:{position}')
                        position = [position[0] * scale[1], position[1] * scale[0], position[2] * scale[2]]
                        position = [position[0] + offset[1], position[1] - offset[0], position[2] + offset[2]]
                    else:
                        # self.get_logger().info(f'3:{position}')
                        position = [position[0] * scale[0], position[1] * scale[1], position[2] * scale[2]]
                        position = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]

                    # self.get_logger().info(f'{position}')
                    finish = pick_and_place.place(position, 80, yaw, 200, self.joints_pub, self.kinematics_client)
                    if finish:
                        self.go_left()
                    else:
                        self.go_home()
                else:
                    self.go_home()
                self.target = None
                self.start_transport = False
            else:
                time.sleep(0.1)

    def main(self):
        while self.running:
            if self.enter:
                try:
                    bgr_image = self.image_queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue

                if self.start_get_roi:
                    self.get_roi()
                    self.start_get_roi = False
                roi = self.roi.copy()
                intrinsic = self.intrinsic
                target_info = []
                if  self.enable_stackup and not self.start_transport:

                    tags = self.at_detector.detect(cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY), True, (intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]), self.tag_size)
                    if len(tags) > 0 :

                        index = 0
                        for tag in tags:
                            if 'tag%d'%tag.tag_id in self.target_labels:
                                corners = tag.corners.astype(int)
                                cv2.drawContours(bgr_image, [corners], -1, (0, 255, 255), 2, cv2.LINE_AA)
                                rect = cv2.minAreaRect(np.array(tag.corners).astype(np.float32))
                                # rect 包含 (中心点, (宽度, 高度), 旋转角度)
                                (center, (width, height), angle) = rect
                                index += 1
                                target_info.append(['tag%d'%tag.tag_id, index, (int(center[0]), int(center[1])), (int(width), int(height)), angle])
                                # self.get_logger().info(f'target_info:{target_info}')

                        if self.get_height:
                            # 获取标签木块的高度
                            self.count += 1
                            if self.count > 15:
                                self.count = 0
                                pose_end = np.matmul(self.hand2cam_tf_matrix, common.xyz_rot_to_mat(tags[0].pose_t, tags[0].pose_R))  # 转换到末端相对坐标(relative coordinates of the converted end)
                                pose_world = np.matmul(self.endpoint, pose_end)  # 转换到机械臂世界坐标(convert to the robotic arm world coordinates)
                                pose_world_T, _ = common.mat_to_xyz_euler(pose_world, degrees=True)
                                if self.camera_type == 'USB_CAM':
                                    pose_world_T[2] += 0.04
                                if pose_world_T[-1] > 0.09:
                                    self.err_msg = "Too high, please remove some blocks first!!!"
                                else:
                                    self.err_msg = None
                                    self.place_position[2] = pose_world_T[2] + 0.01
                                    self.get_height = False
                                    self.go_home()

                    if target_info:
                        if self.last_object_info_list:
                            # 对比上一次的物体的位置来重新排序
                            target_info = position_change_detect.position_reorder(target_info, self.last_object_info_list, 20)
                    self.last_object_info_list = copy.deepcopy(target_info)
                    for target in target_info:
                        cv2.putText(bgr_image, '{}'.format(target[0]),(target[2][0] - 4 * len(target[0] + str(target[1])), target[2][1] + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)              
                    target_miss = True 
                    for target in target_info:  # detect
                        if self.target is not None :  # 如果已经有了目标，其他物体就直接跳过
                            if self.target[0] != target[0] or self.target[1] != target[1]:
                                continue
                            else:
                                target_miss = False
                                self.target = target
                        if self.camera_type == 'USB_CAM':
                            x, y = distortion_inverse_map.undistorted_to_distorted_pixel(target[2][0], target[2][1], self.intrinsic, self.distortion)#将校正后图像的像素点反推回原始（带畸变）图像中的像素点
                            target[2] = (x, y)

                        position, projection_matrix = self.get_object_world_position(target[2], intrinsic, self.extristric, self.white_area_center)
                        result = self.calculate_pick_grasp_yaw(position, target, target_info, intrinsic, projection_matrix)
                        if result is not None and self.target is None:
                            self.target = target
                            break
                
                        if self.last_position is not None and self.target is not None and result is not None and not self.get_height :
                            e_distance = round(math.sqrt(pow(self.last_position[0] - position[0], 2)) + math.sqrt(
                                pow(self.last_position[1] - position[1], 2)), 5)
                            # self.get_logger().info(f'e_distance: {e_distance}')
                            if e_distance <= 0.005:  # 欧式距离小于2mm, 防止物体还在移动时就去夹取了
                                cv2.line(bgr_image, result[1][0], result[1][1], (255, 255, 0), 2, cv2.LINE_AA)
                                self.count_move = 0
                                self.count_still += 1
                            else:
                                self.count_move += 1
                                self.count_still = 0

                            if self.count_move > 10:
                                self.target = None
                            if self.count_still > 20:
                                self.count_still = 0
                                self.count_move = 0
                                # self.get_logger().info(f'pick:{position}')
                                self.target = target
                                yaw = 500 + int(result[0] / 240 * 1000)
                                self.transport_info = [position, yaw, target]
                                self.start_transport = True
                        self.last_position = position
                    if target_miss:
                        self.target_miss_count += 1
                    if self.target_miss_count > 10:
                        self.target_miss_count = 0
                        self.target = None


                if bgr_image is not None and self.get_parameter('display').value:
                    cv2.imshow('result_image', bgr_image)
                    cv2.waitKey(1)
                if self.err_msg is not None:
                    self.get_logger().error(self.err_msg)
                    err_msg = self.err_msg.split(';')
                    for i, m in enumerate(err_msg):
                        cv2.putText(bgr_image, m, (10, 150 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 7)
                        cv2.putText(bgr_image, m, (10, 150 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))
            else:
                time.sleep(0.1)


def main():
    node = TagStackup('tag_stackup')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


