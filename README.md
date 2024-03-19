# Representation Learning for Model-Predictive Exploration in Reinforcement Learning

## Motivation:
It is hard for Reinforcement Learning Agents to learn with high dimensional state spaces. This is why it is important to have smaller state representations. Adding auxiliary losses for representation learning is one way to deal with this. 
Furthermore, exploration is crucial for Reinforcement Learning problems, especially for problems with high dimensional state spaces. There are many approaches which were proposed to deal with this issue. One of these methods is based on "prediction-error". With this method, one predicts "something" (e.g. next state, reward) then we compare it against our actual observations. If the discrepancy between those two is high, one concludes that further exploration of such states is required to decrease the error. It gets harder to perform this with high dimensional spaces for reasons such as curse of dimensionality, and the noise present. 
Following from these two findings, we believe that for "prediction-based" exploration methods, receiving support from representation learning methods appears as a viable solution.

---

### Installation:
_Note: developed using python==3.8.16, pip==23.0.1, ubuntu==22.04.3_
* Installation with Docker:
    ```bash
    make docker_build # create Image
    make docker_start # create and run Container
    ```
    ---

* Installation with conda:
    ```bash
    conda create --name <env> python=3.8.16 --file requirements.txt
    ```
## Usage:

* Training RND only:
    Use a config file with _RepresentationLearningParameter = None_.
* Train RND + BYOL:
    Use a config file with _RepresentationLearningParameter = BYOL_.
* Train RND + Barlow-Twins:
    Use a config file with _RepresentationLearningParameter = Barlow-Twins_.


---

### Torchrun Distributed Training/Testing (Example):

* Train RND agent in MontezumaRevenge from scratch:
    ```bash
    torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/MontezumaRevenge/config_rnd00.conf --log_name=MontezumaRevenge_rnd00 --save_model_path=checkpoints/MontezumaRevenge/rnd00.ckpt
    ```

* Continue Training RND agent from a checkpoint in MontezumaRevenge:
    1. set _loadModel = True_ in the corresponding config file.
    ```bash
    torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/MontezumaRevenge/config_rnd00.conf --log_name=MontezumaRevenge_rnd00_cont00 --save_model_path=checkpoints/MontezumaRevenge/rnd00_cont00.ckpt --load_model_path=checkpoints/MontezumaRevenge/rnd00.ckpt
    ```
* Note that this examples uses 1 node and 128 processes in total. It will have 1 process as agent/trainer and another 127 processes as environment workers. You can modify the parameters of torchrun to suit your needs.

By default, if your machine has more than 1 GPU then automatically a unique GPU will be assigned per trainer process. In order to use a specific GPU, you should pass _--gpu_id=*_ command line parameter and must have _GLOBAL_WORLD_SIZE == 1_.
* Train RND agent in MontezumaRevenge from scratch using only the GPU _cuda:1_:
    ```bash
    torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/MontezumaRevenge/config_rnd00.conf --log_name=MontezumaRevenge_rnd00 --save_model_path=checkpoints/MontezumaRevenge/rnd00.ckpt --gpu_id=1
    ```

---
* Test a RND agent in MontezumaRevenge:
    ```bash
    torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --eval --config_path=./configs/demo_config.conf --log_name=MontezumaRevenge_rnd00 --load_model_path=checkpoints/rnd00.ckpt
    ```
* Note that nproc_per_node has to be 2 for testing of the agent. This is because it supports only a single environment worker during training.

---
### Profiling
* Profiling with Scalene (torchrun Example):
    * Logs are directly outputted to browser:
        ```bash
        python -m scalene --- -m torch.distributed.run --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 3 --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --scalene_profiling 3
        ```
    * Logs are not opened on the browser but are logged to an output file:
        ```bash
        python -m scalene --no-browser --outfile scaleneProfiler00_rnd00.html --- -m torch.distributed.run --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 3 --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --scalene_profiling 3
        ```
        or
        ```bash
        make scalene_profiling
        ```

* Profiling with Pytorch Profiler (torchrun Example):
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 3 --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --pytorch_profiling
    ```
    or
    ```bash
    make pytorch_profiling
    ```
    * To see the profiling results run:
        ```bash
        make start_tensorboard_profiles
        ```

* Profiling with Scalene options example (not suitable for our torchrun code):
    ```bash
    scalene --no-browser --cpu --gpu --memory --outfile profile_rnd_montezuma.html --profile-interval 120 main.py --train --config_path=./configs/demo_config.conf --log_name=rnd00 --save_model_path=checkpoints/rnd00.ckpt
    ```
---

### Some helper commands
* Kill RND code (and its subprocesses):
    ```bash
    make kill
    ```
* Start Tensorboard Server:
    ```bash
    make start_tensorboard
    ```

---

## Overview
* __Configurations__:
    * Most of the running parameters/options are set inside a config file (_*.conf_). These files are located in _./configs_ directory. An explanation of the available options can be found by running:
        ```bash
        python3 main.py --train --config_options
        ```
    * Once you have a config file, you need to provide command line arguments to specify some other options. An explanation of the available command line options can be found by running:
        ```bash
        python3 main.py -h
        ```
    ---

* __Supported OpenAI GYM Environments__
    * Atari (https://www.gymlibrary.dev/environments/atari/index.html):
        * Montezuma Revenge:
            in the config file set:
            ```config
            EnvType = atari
            EnvID = MontezumaRevengeNoFrameskip-v4
            ```
        * Pong:
            in the config file set:
            ```config
            EnvType = atari
            EnvID = PongNoFrameskip-v4
            ```

    * Super Mario Bros (https://pypi.org/project/gym-super-mario-bros/):
        * Super Mario Bros:
            in the config file set:
            ```config
            EnvType = mario
            EnvID = SuperMarioBros-v0
            ```

    * Classic Control (https://www.gymlibrary.dev/environments/classic_control/):
        * Cart Pole:
            in the config file set:
            ```config
            EnvType = classic_control
            EnvID = CartPole-v1
            ```
    ---
- __Distributed Training Architecture__
    * The code relies on _torch.distributed_ package to implement distributed training. It is implemented so that every node is assigned a single agent (GPU) which gathers rollouts by interacting with the environment workers (CPU) and trains the agent. The rest of the processes in a given node are assigned as the environment workers. These processes have an instance of the gym environment and are used solely to interact with these environments in a parallelized manner. Every agent(trainer) process sends sends actions to the environment worker processes in their node and gathers interactions with the environments. Then, these interactions are used to train the model. Gradients across agent workers are synchronized by making use of the _DistributedDataParallel_ module of _torch_.
    
    * In every node, 1 process (process with local_rank == 0) is assigned to the agents_group, the remaining processes are
    assigned to the env_workers_group. To get a better understanding check out the example below.
    agents_group processes have an instance of RNDAgent and perform optimizations.
    env_workers_group processes have an instance of the environment and perform interactions with it.
        ```txt
        Example:

            Available from torchrun:
                nnodes: number of nodes = 3
                nproc_per_node: number of processes per node = 4
            ---

            ************** NODE 0:
            LOCAL_RANK 0: GPUs --> agents_group
            LOCAL_RANK != 0: CPUs --> env_workers_group
            **************
            ...

            ************** NODE: 1:
            LOCAL_RANK 0: GPUs --> agents_group
            LOCAL_RANK != 0: CPUs --> env_workers_group
            **************
            ...

            ************** NODE: 2:
            LOCAL_RANK 0: GPUs --> agents_group
            LOCAL_RANK != 0: CPUs --> env_workers_group
            **************

            -node0-  -node1-   -node2-
            0,1,2,3  4,5,6,7  8,9,10,11    ||    agents_group_ranks=[0,4,8], env_workers_group_rank=[remaining ranks]
            *        *        *
        ```
    ---

* __Tests__
    * _tests.py_: This file contains some tests for environment wrappers and custom environment implementations.
    ```bash
    make run_tests
    ```

---


### Appendix:
__Model Predictive Exploration:__
* Random Network Distillation (RND): 
    
    * Paper: https://arxiv.org/abs/1810.12894
    * Code: https://github.com/jcwleo/random-network-distillation-pytorch

    ---

__Non-Contrastive Representation Learning:__
* BYOL-Explore: Exploration by Bootstrapped Prediction:

    * Paper: https://arxiv.org/abs/2206.08332


* Bootstrap Your Own Latent a New Approach to Self-Supervised Learning (BYOL):

    * Paper: https://arxiv.org/pdf/2006.07733.pdf
    * Code: 
        
        1. https://github.com/The-AI-Summer/byol-cifar10/blob/main/ai_summer_byol_in_cifar10.py#L92
        2. https://github.com/SaeedShurrab/Simple-BYOL/blob/master/byol.py
        3. https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

* Barlow Twins: Self-Supervised Learning via Redundancy Reduction

    * Paper: https://arxiv.org/pdf/2103.03230.pdf
    * Code:

        1. https://github.com/facebookresearch/barlowtwins
        2. https://github.com/MaxLikesMath/Barlow-Twins-Pytorch
