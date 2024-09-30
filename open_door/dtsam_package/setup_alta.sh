# Create a virtual environment.
conda create -n rm python=3.8
conda activate rm

# Install pip dependencies.
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install open3d scikit-image Flask

# Install Detectron2.
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

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
mkdir -p segment-anything/models/
mv sam_vit_h_4b8939.pth segment-anything/models/sam_vit_h_4b8939.pth