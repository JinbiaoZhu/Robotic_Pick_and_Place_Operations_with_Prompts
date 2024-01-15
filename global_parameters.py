"""
Description: This file defines some global parameters.
Editor: Jinbiao Zhu
Date: 15-01-2024
"""


class GlobalParam:
    delta_instance = 0.001
    delta_times = 40
    arm_joints_num = 6
    gripper_force = 20

    OPENAI_API_KEY_for_Agent = "sk-7G6F2kvF7lltG7aGF15b71488d974aDaA8C247C489Cb786c"

    OpenAI_chat_model = "gpt-3.5-turbo"
    OpenAI_response_model = "text-davinci-003"

    Default_temperature = 0.5
    Default_max_tokens = 2560

    up_coordinate = 0.2
    down_coordinate = 0.05

    ROOT_PATH = "/home/zjb/PycharmProjects/RobotGPT/"

    score_threshold = 0.3
