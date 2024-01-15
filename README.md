# Robotic_Pick_and_Place_Operations_with_Prompts

The implementation of the paper "Enhancing home service robot system: Leveraging foundation models to pick-place objects guided by human natural language queries".

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

1. Go to the Coppeliasim download [website](https://www.coppeliarobotics.com/previousVersions).
2. Download `CoppeliaSim Player, Ubuntu 20.04`. It should be a `.tar.xz` file.
3. Extract the `.tar.xz` file to `/home/your-account/`.

### Clone the repository

```
git clone https://github.com/JinbiaoZhu/Robotic_Pick_and_Place_Operations_with_Prompts.git
```

### Download Coppeliasim scene

Download [this file](https://drive.google.com/file/d/1FxXkRcFUu9Og7UsbsiMjtfF2nxXHiBzY/view?usp=drive_link) (`PickPlaceScene.ttt`).

Move this file to `Robotic_Pick_and_Place_Operations_with_Prompts` directory.

### Download the vision models

1. Download [this file](https://drive.google.com/file/d/1HPY5hxVC7AE3T9ZJIcK-gisoQOLEkyFf/view?usp=drive_link) (`used_owlvit_sam.tar.xz`).
2. Extract the `used_owlvit_sam.tar.xz` file to `Robotic_Pick_and_Place_Operations_with_Prompts` directory.

## Configuration

### Install required Python packages




