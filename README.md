# AR-DAE: Towards Unbiased Neural Entropy Gradient Estimation 
Pytorch implementation of AR-DAE on our paper: 
> Jae Hyun Lim, Aaron Courville, Christopher Pal, Chin-Wei Huang, [*AR-DAE: Towards Unbiased Neural Entropy Gradient Estimation*](https://arxiv.org/abs/2006.05164) (2020)

## Toy example of AR-DAE
Example code to train AR-DAE on swiss roll dataset:  
[ipython-notebook](https://github.com/lim0606/pytorch-ardae-vae/tree/master/notebooks/ardae_toy.ipynb)

## Energy function fitting with AR-DAE
Example code to train an implicit sampler using AR-DAE-based entropy gradient estimator:  
[ipython-notebook](https://github.com/lim0606/pytorch-ardae-vae/tree/master/notebooks/ardae_fit.ipynb)

## AR-DAE VAE
please find the code at https://github.com/lim0606/pytorch-ardae-vae

## SAC-AR-DAE

### Getting Started

#### Requirements 
`python>=3.6`  
`pytorch==1.4.0`  
`tensorflow` (for tensorboardX)  
`tensorboardX`  
`git+https://github.com/lim0606/torchkit.git` (for sac-nf) 

##### Requirements (OpenAI gym)
`mujoco`  
`mujoco_py`  

- Install mujoco
  ```
  wget https://mujoco.org/download/mujoco200_linux.zip
  unzip mujoco200_linux.zip
  mkdir -p ~/.mujoco
  mv mujoco200_linux ~/.mujoco/mujoco200
  ```
  
  ```
  cp mjkey.txt ~/.mujoco/.
  ```
  
- Install mujoco_py
  ```
  git clone https://github.com/openai/mujoco-py.git
  cd mujoco-py
  pip install -e .
  ```

- Install gym
  ```
  git clone https://github.com/openai/gym.git
  cd gym
  pip install -e .
  ```

##### Requirements ([Rllab](https://github.com/rll/rllab/blob/master/environment.yml))
`mujoco131`  
`git+https://github.com/inksci/mujoco-py-v0.5.7.git`  
`git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano`  
`git+https://github.com/openai/gym.git@v0.7.4#egg=gym`  
`PyOpenGL`  
`pyglet`  
`rllab`  

- Install rllab
  ```
  git clone https://github.com/rll/rllab.git
  cd rllab
  pip install -e .
  ```

- Install mujoco
  ```
  wget https://mujoco.org/download/mujoco131_linux.zip
  unzip mujoco131_linux.zip
  mv mjpro131 <path-to-rllab>/vendor/mujoco
  ```
  
  ```
  cp mjkey.txt <path-to-rllab>/vendor/mujoco/.
  ```

#### Structure
- `utils`: miscelleneous functions
- `models`: model classes for ar-dae
- `model.py` model classes for rl experiments
- `main_gs.py`: main function to train model (sac) 
- `main_nf.py`: main function to train model (sac-nf) 
- `main_ardae.py`: main function to train model (sac-ar-dae) 

### Experiments
#### Train
- For example, you can train a SAC-AR-DAE for Ant-v2 environment as follows,  
  ```sh
  python main_ardae.py --cuda \
  --cache experiments/ant --env-name Ant-v2 \
  --alpha 0.05 --start_steps 10000 \
  --noise_size 10 --policy_type mlp --policy_nonlin elu --num_enc_layers 1 --num_fc_layers 1 \
  --lmbd 100000 --nu 1.1 --eta 0.01 --num-pert-samples 10 --jac-act tanh \
  --gqnet_nonlin relu --gqnet_num_layers 1 \
  --dae-type grad --dae-nonlin elu --dae_num_layers 5 --dae-enc-ctx true --dae-ctx-type state \
  --train-nz-cdae 128 --train-nstd-cdae 1 --num-cdae-updates 1 \
  --std-scale 10000 --delta 0.1 \
  --d-optimizer adam --d-lr 0.0003 --d-beta1 0.9 --d-momentum 0.9 \
  --q-optimizer adam --lr 0.0003 --q-beta1 0.9 --q-momentum 0.9 \
  --mean-sub-method none --mean-upd-method avg --mean-sub-tau 0.005 --use-ptfnc 100 \
  --log-interval 1000 --eval-interval 10000 --ckpt-interval 20000 --seed -1 --exp-num 1
  ```  
  For more information, please find example scripts, `run.sh`.
  
  
## Contact
For questions and comments, feel free to contact [Jae Hyun Lim](mailto:jae.hyun.lim@umontreal.ca) and [Chin-Wei Huang](mailto:chin-wei.huang@umontreal.ca).

## License
MIT License

## Reference
```
@article{jaehyun2020ardae,
  title={{AR-DAE}: Towards Unbiased Neural Entropy Gradient Estimation},
  author={Jae Hyun Lim and
          Aaron Courville and
          Christopher J. Pal and
          Chin-Wei Huang},
  journal={arXiv preprint arXiv:2006.05164},
  year={2020}
}
```
