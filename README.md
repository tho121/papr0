Follow install instructions for zero123:
https://github.com/cvlab-columbia/zero123

Follow install instructions for papr:
https://github.com/zvict/papr

Install everything to the same environment.

Add pytorch 1.13.1 with cuda 11.7 manually with:

for venv

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

for conda

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia


Run zero123 and papr with:

3drec/run_papr0.py --opt configs/nerfsyn/chair.yml