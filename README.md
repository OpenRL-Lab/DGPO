# DGPO

Wentse Chen, Shiyu Huang, Yuan Chiang, Tim Pearce, Wei-Wei Tu, Ting Chen, Jun Zhu

This is the official code implementation of the paper "DGPO: Discovering Multiple Strategies with Diversity-Guided Policy Optimization"

This repository is heavily based on https://github.com/marlbenchmark/on-policy. 

## Environments supported:

#### Released
- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)

#### TODO
- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)






## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

```
pip install torch matplotlib tensorboardX gym pysc2 absl-py
```

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd on-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.


### 2.1 MPE

``` Bash
# install this package first
pip install seaborn
```

### 2.2 StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.




## 3.Train
Here we use train_mpe.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

## 4. Publication

If you find this repository useful, please cite our [paper](https://arxiv.org/abs/2207.05631):
```
@article{chen2022dgpo,
  title={DGPO: discovering multiple strategies with diversity-guided policy optimization},
  author={Chen, Wenze and Huang, Shiyu and Chiang, Yuan and Pearce, Tim and Tu, Wei-Wei and Chen, Ting and Zhu, Jun},
  journal={arXiv preprint arXiv:2207.05631},
  year={2022}
}
```

