# Robotic_Pick_and_Place_Operations_with_Prompts

The implementation of the paper "**Enhancing home service robot system: Leveraging foundation models to pick-place objects guided by human natural language queries**".

## Preparation

1. System: Ubuntu 20.04
2. CPU:
3. GPU:
4. OpenAI API key: [here](https://openai.com/blog/openai-api).

## Installation

### Create conda environment

```
conda activate
conda create -n pickplace python==3.10.0
conda activate pickplace
```

### Install Coppeliasim

1. Go to the Coppeliasim downloading [website](https://www.coppeliarobotics.com/previousVersions).
2. Download `CoppeliaSim Player, Ubuntu 20.04`. It should be a `.tar.xz` file.
3. Extract the `.tar.xz` file to `/home/your-account/`.

### Clone the repository

```
git clone https://github.com/JinbiaoZhu/Robotic_Pick_and_Place_Operations_with_Prompts.git
```

### Download Coppeliasim scene

1. Download [this file](https://drive.google.com/file/d/1FxXkRcFUu9Og7UsbsiMjtfF2nxXHiBzY/view?usp=drive_link) (`PickPlaceScene.ttt`) from Google drive.
2. Move this file to `Robotic_Pick_and_Place_Operations_with_Prompts` directory.

### Download the vision models

1. Download [this file](https://drive.google.com/file/d/1HPY5hxVC7AE3T9ZJIcK-gisoQOLEkyFf/view?usp=drive_link) (`used_owlvit_sam.tar.xz`) from Google drive.
2. Extract the `used_owlvit_sam.tar.xz` file to `Robotic_Pick_and_Place_Operations_with_Prompts` directory.

## Configuration

### Install required Python packages

```
conda activate pickplace
cd Robotic_Pick_and_Place_Operations_with_Prompts
pip install -r requirements.txtsave 
```

### Enter Coppeliasim and load PickAndPlace.ttt scene

Launch another terminal to enter Coppeliasim.

```
cd CoppeliaSim_Player_V4_1_0_Ubuntu20_04/
./coppeliasim.sh
```

`File` -- `Open scene...` -- `/home/your-account/Robotic_Pick_and_Place_Operations_with_Prompts/` -- select the `PickPlaceScene.ttt`.

### Config parameters

Open the file `global_parameters.py`, select `OPENAI_API_KEY_for_Agent`, `OpenAI_chat_model`, `OpenAI_response_model`, `ROOT_PATH` and save this file.

## Launch project

Open Coppeliasim window, and click `Start` bottom.
Then run `test_proposed_current.py` script.


