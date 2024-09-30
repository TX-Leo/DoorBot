# Create a virtual environment.
python -m venv venv
source venv/bin/activate

# Install pip dependencies.
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install open3d scikit-image Flask
# python -m pip install pillow==9.5.0 ＃ for debuging: PIL.Image.LINEAR no longer exists

# Install Detectron2.
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
# it works if the detectron2.utils is not found

# Clone and install Detic.
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt
cd ../

# Clone and install Segment Anything.
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ../

# Download SAM checkpoint.
# If this link does not work, check https://github.com/facebookresearch/segment-anything.
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
