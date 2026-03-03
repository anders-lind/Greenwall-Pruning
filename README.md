1. Clone repository
``` bash
git clone https://github.com/alexellegaard/Greenwall-Pruning.git
```

2. Install dynamixel sdk (IMPORTANT: Must not be inside a python venv)
``` bash
pip install dynamixel_sdk --break-system-packages
```

3. Give access do dynamixel motors (/ttyUSB0 for the most part)
``` bash
sudo chmod a+rw /dev/ttyUSB*
```

4. Install SAM2 for image segmentation \
Follow This install guide: https://github.com/facebookresearch/sam2?tab=readme-ov-file