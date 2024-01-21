"""
Description: This file defines some global parameters.
Editor: Jinbiao Zhu
Date: 15-01-2024
"""


class GlobalParam:

    # [Attention] Put you Openai api-key here
    OPENAI_API_KEY_for_Agent = "sk-xxxxxxxxxx"

    # [Attention] Change this ROOT_PATH here!
    ROOT_PATH = "/home/name/directory/"
    
    delta_instance = 0.001
    delta_times = 40
    arm_joints_num = 6
    gripper_force = 20
    OpenAI_chat_model = "gpt-3.5-turbo"
    OpenAI_response_model = "text-davinci-003"
    Default_temperature = 0.5
    Default_max_tokens = 2560
    up_coordinate = 0.2
    down_coordinate = 0.05
    score_threshold = 0.3
