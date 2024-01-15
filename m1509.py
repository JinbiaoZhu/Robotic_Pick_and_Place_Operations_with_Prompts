"""
Description: This is Doosan M1509 Robot's python codes in Coppeliasim/Vrep.
Editor: Jinbiao Zhu
Date: 15-01-2024
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sim
from global_parameters import GlobalParam
from owlvit_segment_anything.owl_sam_func import OwlvitSAM
from skillset import *
from toolset import *
from utils.tools import get_current_date_and_time


class M1509v2(Skills, Tools):
    def __init__(self):
        """
        Connect to a running Coppeliasim simulation environment.
        """
        super(Skills).__init__()
        super(Tools).__init__()
        self.global_file_path = GlobalParam.ROOT_PATH + "image_records/" + get_current_date_and_time() + ".png"
        self.global_depth_path = GlobalParam.ROOT_PATH + "image_records/" + get_current_date_and_time() + "depth" + ".png"

        # Close all connections first in case
        sim.simxFinish(-1)

        # Connect to a running Coppeliasim.
        clientID = sim.simxStart('127.0.0.1', 19997,
                                 True, True,
                                 5000, 5)
        print(f'Connecting to remote API server using client ID: {clientID}...')

        # Check if the data can be retrieved from the simulation environment
        res, _ = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print('Connecting successfully!')
            pass
        else:
            print('Remote API function call returned with error code: ', res)

        # Initialize global parameters within the class
        self.clientID = clientID

        # An important point for controlling the movement of the robotic arm!
        self.tcp_name = "Target"
        _, self.tcp_handle = sim.simxGetObjectHandle(self.clientID, self.tcp_name, sim.simx_opmode_blocking)

        # Located in the center of the simulation environment and used for coordinate reference!
        self.world_name = "Center"
        _, self.world_handle = sim.simxGetObjectHandle(self.clientID, self.world_name, sim.simx_opmode_blocking)

        # Visual reference point!
        self.VisionPointLeft = "VisionPointLeft"
        _, self.VisionPointLeft_handle = sim.simxGetObjectHandle(self.clientID, self.VisionPointLeft,
                                                                 sim.simx_opmode_blocking)

        # the gripper control script
        self.gripper_name = 'RG2_open'

        # 6-axis
        self.joints_num = GlobalParam.arm_joints_num

        # Obtain the initial pose of the end of the robotic arm
        home_name = "HomePoint"
        _, init_handle = sim.simxGetObjectHandle(self.clientID, home_name, sim.simx_opmode_blocking)
        _, self.init_position = sim.simxGetObjectPosition(self.clientID, init_handle,
                                                          -1, sim.simx_opmode_blocking)
        _, self.init_orientation = sim.simxGetObjectOrientation(self.clientID, init_handle,
                                                                -1, sim.simx_opmode_blocking)

        # Initialize robot arm closing
        self.grasp(True)

        # -------------------------------------------------------------------------------------------------
        # Import the open vocabulary object detector owlvit-SAM
        self.ovd = OwlvitSAM(get_topk=True,
                             owlvit_model="owlvit_segment_anything/owlvit-large-patch14",
                             sam_model="owlvit_segment_anything/segment_anything/sam_vit_h_4b8939.pth")
        self.ovd_outputs = None
        self.pixels = 394
        self.scene_length = 0.75  # unit: meter
        self.scene_depth = 0
        self.rate = self.scene_length / self.pixels

    def kill(self):
        """
        Deprecated.
        """
        print("Killing the connection...")
        sim.simxGetPingTime(self.clientID)
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
        sim.simxFinish(self.clientID)
        print("Killed.")

    def move_horizontally(self, object_name):
        """
        Controls the horizontal movement of the end-effector on the plane where the current position is located.
        """
        # Check whether the object exists.
        if object_name in [output['label'] for output in self.ovd_outputs]:

            # Extract the bounding box of an object
            for item in self.ovd_outputs:
                if item['label'] == object_name:
                    itembox, itemmask = item['box'], item['mask']
                    break
                else:
                    itembox, itemmask = None, None

            object_position = self._calculate_xyxy_to_coordinate(itembox, itemmask)

            # Get the current 3D position of the robotic arm's end-effector
            _, tcp_position = sim.simxGetObjectPosition(self.clientID, self.tcp_handle, -1, sim.simx_opmode_blocking)

            # Set the coordinate change amount of one-step motion according to global parameters
            delta_x = (object_position[0] - tcp_position[0]) / GlobalParam.delta_times
            delta_y = (object_position[1] - tcp_position[1]) / GlobalParam.delta_times

            # Start moving
            for timestep in range(GlobalParam.delta_times):
                tcp_position[0] += delta_x
                tcp_position[1] += delta_y
                sim.simxSetObjectPosition(self.clientID, self.tcp_handle, -1, tcp_position, sim.simx_opmode_blocking)

            pass

        elif object_name == "Alice":
            _, Alice_handle = sim.simxGetObjectHandle(self.clientID, "Alice", sim.simx_opmode_blocking)
            _, Alice_position = sim.simxGetObjectPosition(self.clientID, Alice_handle, -1, sim.simx_opmode_blocking)

            # Get the current 3D position of the robotic arm's end-effector
            _, tcp_position = sim.simxGetObjectPosition(self.clientID, self.tcp_handle, -1, sim.simx_opmode_blocking)

            # Set the coordinate change amount of one-step motion according to global parameters
            delta_x = (Alice_position[0] - tcp_position[0]) / GlobalParam.delta_times
            delta_y = (Alice_position[1] - tcp_position[1]) / GlobalParam.delta_times

            # Start moving
            for timestep in range(GlobalParam.delta_times):
                tcp_position[0] += delta_x
                tcp_position[1] += delta_y
                sim.simxSetObjectPosition(self.clientID, self.tcp_handle, -1, tcp_position, sim.simx_opmode_blocking)

            pass

        else:
            print(f"Move failed because no the {object_name} detected!!!")

    def move_vertically(self, object_name):
        """
        Control the robotic arm's end-effector to move up and down in the vertical direction of the current position.
        """
        if object_name == "Up":
            # Get the current 3D position of the robotic arm's end-effector
            _, tcp_position = sim.simxGetObjectPosition(self.clientID, self.tcp_handle, -1, sim.simx_opmode_blocking)

            # Set the coordinate change amount of one-step motion according to global parameters
            delta_z = (GlobalParam.up_coordinate - tcp_position[2]) / GlobalParam.delta_times

            # Start moving
            for timestep in range(int(GlobalParam.delta_times)):
                tcp_position[2] += delta_z
                sim.simxSetObjectPosition(self.clientID, self.tcp_handle, -1, tcp_position, sim.simx_opmode_blocking)

            pass

        elif object_name == "Down":
            # Get the current 3D position of the robotic arm's end-effector
            _, tcp_position = sim.simxGetObjectPosition(self.clientID, self.tcp_handle, -1, sim.simx_opmode_blocking)

            # Set the coordinate change amount of one-step motion according to global parameters
            delta_z = (GlobalParam.down_coordinate - tcp_position[2]) / GlobalParam.delta_times

            # Start moving
            for timestep in range(int(GlobalParam.delta_times)):
                tcp_position[2] += delta_z
                sim.simxSetObjectPosition(self.clientID, self.tcp_handle, -1, tcp_position, sim.simx_opmode_blocking)

            pass

        elif object_name == "Alice":
            _, Alice_handle = sim.simxGetObjectHandle(self.clientID, "Alice", sim.simx_opmode_blocking)
            _, Alice_position = sim.simxGetObjectPosition(self.clientID, Alice_handle, -1, sim.simx_opmode_blocking)

            # Get the current 3D position of the robotic arm's end-effector
            _, tcp_position = sim.simxGetObjectPosition(self.clientID, self.tcp_handle, -1, sim.simx_opmode_blocking)

            # Set the coordinate change amount of one-step motion according to global parameters
            delta_z = (Alice_position[2] - tcp_position[2]) / GlobalParam.delta_times

            # Start moving
            for timestep in range(int(GlobalParam.delta_times)):
                tcp_position[2] += delta_z
                sim.simxSetObjectPosition(self.clientID, self.tcp_handle, -1, tcp_position,
                                          sim.simx_opmode_blocking)

            pass

        else:
            # Check whether the object exists.
            if object_name in [output['label'] for output in self.ovd_outputs]:

                # Extract the bounding box of an object
                for item in self.ovd_outputs:
                    if item['label'] == object_name:
                        itembox, itemmask = item['box'], item['mask']
                        break
                    else:
                        itembox, itemmask = None, None

                object_position = self._calculate_xyxy_to_coordinate(itembox, itemmask)

                # Get the current 3D position of the robotic arm's end-effector
                _, tcp_position = sim.simxGetObjectPosition(self.clientID, self.tcp_handle, -1,
                                                            sim.simx_opmode_blocking)

                # Set the coordinate change amount of one-step motion according to global parameters
                delta_z = (object_position[2] - tcp_position[2]) / GlobalParam.delta_times

                # Start moving
                for timestep in range(int(GlobalParam.delta_times)):
                    tcp_position[2] += delta_z
                    sim.simxSetObjectPosition(self.clientID, self.tcp_handle, -1, tcp_position,
                                              sim.simx_opmode_blocking)

                print("Move OK!!!")

            else:
                print(f"Move failed because no the {object_name} detected!!!")

    def grasp(self, onoff: bool):
        """
        :param onoff: False = Open = 1; True = Closure = 0
        """
        if onoff:
            # Gripper closure
            sim.simxSetIntegerSignal(self.clientID, self.gripper_name, 0, sim.simx_opmode_blocking)
            time.sleep(0.5)
        else:
            # Gripper open
            sim.simxSetIntegerSignal(self.clientID, self.gripper_name, 1, sim.simx_opmode_blocking)
            time.sleep(0.5)

        # Wait for gripper
        time.sleep(1)

    def home(self):
        """
        Defaults the initial position of the robot arm.
        """
        # Get the current 3D position of the end of the robotic arm
        _, current_position = sim.simxGetObjectPosition(self.clientID, self.tcp_handle, -1, sim.simx_opmode_blocking)
        _, current_orientation = sim.simxGetObjectOrientation(self.clientID, self.tcp_handle, -1,
                                                              sim.simx_opmode_blocking)

        # Set the coordinate change amount of one-step motion according to the global parameters
        delta_x = (self.init_position[0] - current_position[0]) / GlobalParam.delta_times
        delta_y = (self.init_position[1] - current_position[1]) / GlobalParam.delta_times
        delta_z = (self.init_position[2] - current_position[2]) / GlobalParam.delta_times
        delta_r = (self.init_orientation[0] - current_orientation[0]) / GlobalParam.delta_times
        delta_p = (self.init_orientation[1] - current_orientation[1]) / GlobalParam.delta_times
        delta_l = (self.init_orientation[2] - current_orientation[2]) / GlobalParam.delta_times

        # Start moving
        for timestep in range(GlobalParam.delta_times):
            current_position[0] += delta_x
            current_position[1] += delta_y
            current_position[2] += delta_z

            sim.simxSetObjectPosition(self.clientID, self.tcp_handle, -1, current_position, sim.simx_opmode_blocking)

        for timestep in range(GlobalParam.delta_times):
            current_orientation[0] += delta_r
            current_orientation[1] += delta_p
            current_orientation[2] += delta_l
            sim.simxSetObjectOrientation(self.clientID, self.tcp_handle, -1,
                                         current_orientation, sim.simx_opmode_blocking)

        pass

    def observe(self, query: str):
        """
        Obtain object information in the scene.
        """
        self._capture()
        self.ovd_outputs = self.ovd.inference(image_path=self.global_file_path,
                                              prompt=query,
                                              need_SAM=True,
                                              need_grounding_results=False,
                                              need_show=False)
        pass

    def _capture(self):
        """
        Capture graphics of sensors in the scene.
        """
        # Get the handle of vision sensor
        _, visionSensorHandle = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor',
                                                        sim.simx_opmode_blocking)
        # Get the handle of depth sensor
        _, depthSensorHandle = sim.simxGetObjectHandle(self.clientID, 'Depth_sensor',
                                                       sim.simx_opmode_blocking)
        # Get the image of vision sensor
        _, resolution, image = sim.simxGetVisionSensorImage(self.clientID, visionSensorHandle, 0,
                                                            sim.simx_opmode_blocking)
        _, depth_resolution, buffer = sim.simxGetVisionSensorDepthBuffer(self.clientID, depthSensorHandle,
                                                                         sim.simx_opmode_blocking)

        sensorImage = np.array(image).astype(np.uint8)
        sensorImage.resize([resolution[0], resolution[1], 3])
        sensorImage = np.flip(sensorImage, 0)

        depthImage = np.array(buffer)
        depthImage.resize([resolution[0], resolution[1]])
        depthImage = np.flip(depthImage, 0)

        self.scene_depth = depthImage.max()

        plt.figure(figsize=(512 / 100, 512 / 100), dpi=100)
        plt.axis("off")
        plt.imshow(sensorImage)
        plt.savefig(self.global_file_path,
                    bbox_inches='tight', pad_inches=0)
        plt.imshow(depthImage)
        plt.savefig(self.global_depth_path,
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    def _calculate_xyxy_to_coordinate(self, box, mask):

        if box is None or mask is None or self.scene_depth == 0:
            print("Error!!!")

        else:

            # Get position based on visual reference node
            _, VisionPointLeft_position = sim.simxGetObjectPosition(self.clientID,
                                                                    self.VisionPointLeft_handle, -1,
                                                                    sim.simx_opmode_blocking)

            # Calculates the coordinates of all pixels in the mask that are True
            indices = np.argwhere(mask)

            # Calculates the center coordinates of the mask
            x_center = int(np.mean(indices[:, 1]))
            y_center = int(np.mean(indices[:, 0]))

            # Load the depth map first and convert it to numpy format
            depth_numpy = np.array(Image.open(self.global_depth_path).convert("L"))

            # Multiply the input mask to obtain the local depth information
            local_depth = depth_numpy * mask

            # Calculate the average depth of the mask section
            mean_depth = np.mean(local_depth[np.nonzero(local_depth)])

            # Calculates the distance from the center point to the visual reference point
            x_distance = x_center * self.rate
            y_distance = y_center * self.rate
            y_coordinate = VisionPointLeft_position[1] + x_distance
            x_coordinate = VisionPointLeft_position[0] + y_distance

            z_coordinate = (self.scene_depth / 2) * (1 - mean_depth / (depth_numpy.max() - depth_numpy.min()))
            z_coordinate *= 0.3

            return [x_coordinate, y_coordinate, z_coordinate]
