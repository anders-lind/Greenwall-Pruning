# Greenwall Pruning

This was the final project for my Master of Science (MsC) in Engieering (Robot Systems) and was a collaboration between myself and Alex Ellegaard (https://github.com/alexellegaard/).

This project designed and implemented a Greenwall pruning robot, which was able to travel around 1m X 1m area on a Greenwall and autonemously prune rotten leaves. The robot was consisted of a cable driven parallel robot (CDPR) with a specialized 3d printed gripper. To find the leaves color-based computer vision was used along with a segmentation model (SAM2).

## Setup

1. Clone this repository
`git clone https://github.com/alexellegaard/Greenwall-Pruning.git`

1. Install Dynamixel SDK (IMPORTANT: Must not be inside a python virtual environment)
`pip install dynamixel_sdk --break-system-packages`

1. Give access to dynamixel motors (/ttyUSB0 for the most part)
`sudo chmod a+rw /dev/ttyUSB*`

1. Install SAM2 for image segmentation
Follow This install guide: https://github.com/facebookresearch/sam2?tab=readme-ov-file
