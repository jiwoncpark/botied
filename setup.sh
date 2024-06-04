# Create conda env
ml awscli
conda update -n base -c defaults conda
conda env create -f env.yaml
conda activate copulala
pip install -e . -r requirements.txt  # TODO: put packages here

# Manually install stable up-to-date torch version (1.13.0, cuda 11.7)
pip install torch torchvision torchaudio psutil
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric
pip install botorch gpytorch
pip install setuptools setuptools_scm pybind11

# Install pyvinecopulib
conda install -c conda-forge gxx_linux-64==11.2.0
ln -s $CC gcc
ln -s $CPP g++
pip install --verbose "git+https://github.com/vinecopulib/pyvinecopulib#egg=pyvinecopulib"
