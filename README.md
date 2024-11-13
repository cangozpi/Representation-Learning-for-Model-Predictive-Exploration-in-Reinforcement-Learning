# Investigating the Effects of Representation Learning on Exploration in On-Policy Reinforcement Learning

## Motivation (Modified RND):
In light of the observation that operating under small-dimensional state spaces is
desirable for both the reinforcement learning agents and the exploration methods,
we believe that for prediction-error based exploration methods, receiving support
from representation learning methods appears as a viable solution. To this end we
propose the Modified RND approach to investigate the effect of using an auxiliary
self-supervised learning (SSL) loss for the model-predictive exploration methods.

The motivation behind the Modified RND approach is as follows. Exploration is an
important concept in RL and it becomes even more important when working with
environments with high-dimensional state spaces. Under such conditions, RND faces
certain issues. Networks in RND gets larger. Potentially, the noise in the data makes
it harder for RND to detect novel states. Also, because the target network in RND
is frozen by design, it might struggle to extract useful features which could result in
RND not being able to distinguish novel states well. Similarly, the PPO’s backbone
gets larger, and it might get harder for the PPO to learn useful image representations
from just reward signals. To this end, our proposed approach promises the following
improvements. In the proposed approach, the RND networks can be smaller since
they operate on features extracted by another network (i.e. PPO’s backbone) and
not the original raw observations. RND can distinguish novel states better due
to it taking useful learned representations as inputs which could result in better exploration. RND’s online network can learn quicker due to operating on already
extracted useful image features. Learning useful embeddings whose size can be tuned
as a hyperparameter is owed to the use of SSL. Furthermore, due to the sharing of the
PPO backbone between RND and PPO, PPO should be able to learn better image
representations in a shorter amount of time because of the introduced auxiliary SSL
loss. This could result in better performing actor and critic heads and overall learn
a better policy. Furthermore, in the original RND paper, RND networks only take
the last frame (i.e. most recent frame of the stacked frames) as input. This could
result in RND not being able to fully understand certain situations. For example,
from a single frame where the agent appears is in mid air, it cannot infer whether
the agent is going to go up in the next frame or if the agent is going to fall down.
On the other hand, in our proposed approach, RND takes an embedding which was
extracted from the stacked frames and hence it potentially encodes the required
temporal information to know what will happen next. For example, assume that
the last four consecutive frames were as given in Figure. Then the Original RND
approach would take the last frame at t + 3 as input, in which the agent appears in
mid air. It is not possible to understand the direction in which the agent has jumped
to appear in such a state, and hence it cannot infer which direction the agent will
move in the next frame. On the other hand, the Modified RND approach operates
on the last four stacked frames. By looking at t+2 and t+3 in Figure, it can
possibly determine that the agent has jumped to the right, which helps clarify the
position the agent will appear in the next frame.

<img src="./assets/mid air example.png" width="80%" title="An example of four consecutive stacked frames where the agent appears
in mid air.">

Figure: An example of four consecutive stacked frames where the agent appears
in mid air.

__Model Architecture:__

The Modified RND approach builds up on the RND approach and modifies it to bring
in the representation learning algorithms in order to reap their potential benefits.
At a high level, it can be thought of as being comprised of the following parts:

* Reinforcement Learning Method:
    * Proximal Policy Optimization (PPO)
* Exploration Method:
    * Random Network Distillation (RND)
* Representation Learning Method:
    * Bootstrap Your Own Latent (BYOL)
    * Barlow Twins

<img src="./assets/modified RND overview byol.png" width="60%" title="Overview of the Modified RND approach using BYOL for SSL.">
<img src="./assets/modified rnd overview barlow.png" width="60%" title="Overview of the Modified RND approach using Barlow-Twins for SSL.">

Figure: Modified RND approach with different SSL methods employed in the
SSL module.


<img src="./assets/shared PPO backbone architecture.png" width="60%" title="Architecture of the shared PPO backbone.">

Figure: Architecture of the shared PPO backbone with a linear last layer.

__Loss Formulation:__
In our approach we are optimizing all of its objectives simultaneously. This is done
by optimizing on a loss function which is a linear combination of the individual
losses as follows:

$L = \alpha_{RL} * \mathcal{L_{\textrm{RL}}} + \alpha_{SSL} * \mathcal{L_{\textrm{SSL}}} + \alpha_{RND} * \mathcal{L_{\textrm{RND}}}$ <br>
$\bullet\notag\textrm{ where RL loss is:}$ <br>
$\qquad\mathcal{L_{\textrm{RL}}} = \alpha_{act} * \mathcal{L_{\textrm{act}}} + \alpha_{crt} * \mathcal{L_{\textrm{crt}}} - \alpha_{\mathcal{H}} * \mathcal{H} \textrm{ and }$ <br>
$\qquad\mathcal{L_{\textrm{crt}}} = \alpha_{crt^{(e)}} * \mathcal{L_{\textrm{crt}^{\textrm{(e)}}}} + \alpha_{crt^{(i)}} * \mathcal{L_{\textrm{crt}^{\textrm{(i)}}}}$

| **Loss Term**        | **Description**                                         |
|----------------------|---------------------------------------------------------|
| $L$             | Total combined loss                                    |
| $\mathcal{L_{\textrm{RL}}}$                 | RL loss (PPO)                                           |
| $\mathcal{L_{\textrm{SSL}}}$                | SSL loss                                                |
| $\mathcal{L_{\textrm{RND}}}$                | RND loss (Random Network Distillation)                  |
| $\mathcal{L_{\textrm{act}}}$                | Actor loss (policy gradient)                            |
| $\mathcal{L_{\textrm{crt}}}$                | Total critic loss (value function)                      |
| $\mathcal{L_{\textrm{crt}^{\textrm{(e)}}}}$             | Critic loss from extrinsic rewards                      |
| $\mathcal{L_{\textrm{crt}^{\textrm{(i)}}}}$             | Critic loss from intrinsic rewards                      |
| $\mathcal{H}$                    | Entropy loss (encourages exploration)                   |
| $\alpha$                    | Coefficients used for linearly weighting the terms      |

## Motivation (Gradient Projection):
It was motivated by the idea that optimizing multiple objectives at the same time
can give rise to certain problems. The objectives might not be aligned leading to
unstable and inferior training performance. The respective loss weights of the ob-
jectives would have to be tuned carefully. Given that one of the objectives is an
auxiliary objective designed to improve and complement the main objective, one
might want to ensure that the auxiliary objective does not hinder the learning of
the main objective.

We believe that training updates which cause the auxiliary objective to hinder the
main objective’s learning can be detected by inspecting their gradients. The gra-
dients can be viewed as two vectors of the same size. If the gradient vector of the
auxiliary objective is pointing in the opposite direction to the gradient vector of the
main objective, one can conclude that such an update would cause the auxiliary
objective to clash with the main objective. Under such conditions, one can choose to project the gradient of the auxiliary objective orthogonal to the main objective’s
gradient and use the projected orthogonal gradient in place of the original gradient
of the auxiliary objective for parameter updates (see the Figure below). This is the idea
behind our proposed gradient projection method, which is based on the notion that
accumulating gradient vectors with components pointing in the opposite directions
would hinder their learning.

In our investigation, the main objective corresponds to the RL objective (PPO
loss) and the auxiliary objective corresponds to the SSL objective (BYOL or Barlow
Twins). We thought that the gradient projection method would help if the gradients
of the PPO loss and the SSL loss happened to clash during training, allowing the
PPO objective to improve even if the auxiliary SSL objective yielded gradients that
would otherwise deteriorate the performance of the PPO objective. One can see how
the gradient projection method might help with optimizing the main objective, given
the hypothetical scenario with the computed gradients and the loss surface presented
in the Figure below. Also, the use of gradient projection could improve the combined
loss function’s tolerance to the weight of the auxiliary loss, as the gradients of the
auxiliary loss that clash with the main objective would be projected orthogonally.
Overall, we believe that using the proposed gradient projection method could lead
to more stable training and better performance in training schemes where different
losses are combined.

__Projection Formula:__

$\textrm{Let } \textbf{a} = \frac{\partial{\mathcal{L}_{SSL}}}{\partial{x}}$

$\textrm{and } \textbf{b} = \frac{\partial{\mathcal{L}_{RL}}}{\partial{x}} \textrm{, }$

$\cos{\theta} = \frac{\textbf{a} \cdot \textbf{b}}{\lVert{\textbf{a}}\rVert \lVert{\textbf{b}}\rVert}$ <br>

$\textit{proj}_{\textbf{b}}\textbf{a} = \frac{\textbf{a} \cdot \textbf{b}}{\lVert{\textbf{b}}\rVert^2} \textbf{b}$

$\text{ and }$ 

$\frac{\partial{\overline{\mathcal{L}_{SSL}}}}{\partial{x}} =$

$\qquad\qquad\textbf{b} - \textit{proj}_{\textbf{b}}\textbf{a}$ <br>

$\notag\text{where } \frac{\partial{\overline{\mathcal{L}_{SSL}}}}{\partial{x}} \text{corresponds to the orthogonally projected gradient vector.}$


<img src="./assets/auxiliary grad projection visualization.png" width="30%" title="auxiliary grad projection visualization.">
<img src="./assets/auxiliary grad not projected visualization.png" width="30%" title="auxiliary grad not projected visualization.">

Figure: The arrows correspond to the gradient vectors. The red arrow corre-
sponds to the projected gradient of the auxiliary objective. The dashed blue arrow
corresponds to the final accumulated gradient (i.e. combination of the main objec-
tive’s gradient and the (un)projected auxiliary objective’s gradient). Image on the
left: the auxiliary SSL objective’s gradient is pointing in the opposite direction so it
is projected orthogonal to the main RL objective. Image on the right: the auxiliary
SSL objective’s gradient is pointing in the same direction to the main RL objective
so its gradients are kept the same.


<img src="./assets/grad proj 2D visualizaiton.png" width="100%" title="Visualization of the gradient projection method in 2D space.">

Figure: Visualization of the gradient projection method in 2D space. The ellipses
depict the loss surface and the arrows correspond to the gradient vectors.

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

## Running:
* __Configurations__:
    * Most of the running parameters/options are set inside a config file (_*.conf_). These files are located in _./configs_ directory. An explanation of the available options can be found by running:
        ```bash
        python3 main.py --train --config_options
        ```
    * Once you have a config file, you need to provide command line arguments to specify some other options. An explanation of the available command line options can be found by running:
        ```bash
        python3 main.py -h
        ```

    See the available SLURM job scrips under _configs/*_ along with the _Overview of the Experiments_ section below.

## Overview of the Experiments:
* __exp21:__ <br>
&emsp;&emsp;    env: Montezuma's Revenge, </br>
&emsp;&emsp;    embedding_dim: [2048]: </br>
&emsp;&emsp;        &emsp; Original RND </br>
&emsp;&emsp;        &emsp; Modified RND (NoSSL) </br>
&emsp;&emsp;        &emsp; JustPPO(NoSSL) </br>
&emsp;&emsp;    Modified RND (with SSL -> BYOL) </br>
&emsp;&emsp;	    &emsp; embedding_dim: [448] </br>
&emsp;&emsp;    Original RND </br>

    ---
* __expGlados2, exp24:__ <br>
&emsp;&emsp;    env: Montezuma's Revenge, </br>
&emsp;&emsp;    embedding_dim: [2048, 448]: </br>
&emsp;&emsp;        &emsp; Modified RND (with SSL -> BYOL) with GradientProjection </br>

    ---
* __exp19, exp20:__ <br>
&emsp;&emsp;    env: [Pong, Breakout], </br>
&emsp;&emsp;    embedding_dim: [2048, 16]: </br>
&emsp;&emsp;        &emsp; JustPPO (NoSSL) </br>
&emsp;&emsp;        &emsp; JustPPO (with SSL -> BYOL) </br>

    ---
* __[expShodan1, expShodan2] == [exp22,exp23]:__ <br>
&emsp;&emsp;    env: [Pong, Breakout], </br>
&emsp;&emsp;    embedding_dim: [2048, 16]: </br>
&emsp;&emsp;        &emsp; JustPPO (with SSL -> BYOL) with GradientProjection </br>

    ---
* __expShodan3, expShodan3-hpc__ <br>
&emsp;&emsp;    env: Montezuma's Revenge, </br>
&emsp;&emsp;    embedding_dim: [2048]: </br>
&emsp;&emsp;        &emsp; Modified RND (with SSL -> BYOL) with PreTraining </br>

    ---
* __expShodan4, expShodan4-hpc__ <br>
&emsp;&emsp;    env: [Pong, Breakout], </br>
&emsp;&emsp;    embedding_dim: [2048, 16]: </br>
&emsp;&emsp;        &emsp; JustPPO (with SSL -> BYOL) with PreTraining </br>

    ---
* __exp25__ <br>
&emsp;&emsp;    env: Montezuma's Revenge, </br>
&emsp;&emsp;    embedding_dim: [448]: </br>
&emsp;&emsp;        &emsp; Modified RND (with SSL -> BYOL) with shared_ppo_backbone_last_layer_type=Conv </br>
&emsp;&emsp;        &emsp; Modified RND (NoSSL) with shared_ppo_backbone_last_layer_type=Conv </br>

    ---
* __exp26__ <br>
&emsp;&emsp;    Modified RND (with SSL -> Barlow Twins) </br>
&emsp;&emsp;    representation coefficient= [10,  1, 0. 1, 0. 001, 0.0001] </br>

    ---
* __expShodan3extension__ <br>
&emsp;&emsp;    Modified RND (with SSL -> Barlow Twins) </br>
&emsp;&emsp;    SSL with backbone unfrozen </br>
&emsp;&emsp;        &emsp; only RL loss </br>
&emsp;&emsp;        &emsp; RL+SSL </br>

    ---
* __exp28__ <br>
 &emsp;&emsp;   Experimented with maintaining two separate copies of the shared PPO backbone. One copy is trained conventionally using both RL and SSL objectives, while the other is updated every n steps—optionally employing Polyak averaging. This second network is used to generate the stationary inputs for the RND Module, somewhat resembling the target networks in DQN.

    ---

---

### Torchrun Distributed Training/Testing (Example):

* __Single Node Single Process with _--num_env_per_process_ many parallel environment processes:__

    * Train RND agent in Montezuma's Revenge from scratch:
        ```bash
        torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
        ```
* __Single Node 2 Processes with each having _--num_env_per_process_ many parallel environment processes (i.e. a total of 2*64=128 many environment processes):__
    * Train RND agent in Montezuma's Revenge from scratch:
        ```bash
        torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
        ```
* __2 Node 2 Processes per node with each having _--num_env_per_process_ many parallel environment processes (i.e. a total of 2*2*64=256 many environment processes):__
    * Train RND agent in Montezuma's Revenge from scratch:
        in __Node1 (assume Node1 has ip 172.20.31.84)__ run:
        ```bash
        torchrun --nnodes 2 --nproc_per_node 2 --rdzv-backend=c10d --rdzv-endpoint=172.20.31.84:33333 --rdzv-id=13 --master-addr=172.20.31.84 main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
        ```
        and in __Node2__ run:
        ```bash
        torchrun --nnodes 2 --nproc_per_node 2 --rdzv-backend=c10d --rdzv-endpoint=172.20.31.84:33333 --rdzv-id=13 --master-addr=172.20.31.84 main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
        ```
        Note: with this approach in Node 1 gpus:['cuda:0', 'cuda:1'] will be used and in Node 2 gpus: ['cuda:0', 'cuda:1'] will be used. There will be a total of 4 agent processes (total of 4 copies of the agent model (i.e. trainable neural network)) and each of them will calculate gradients using rollouts collected from 64 parallel environments, and then distributed package will aggregate the individual computed gradients.

* Continue Training RND agent from a checkpoint in Montezuma's Revenge:
    1. set _loadModel = True_ in the corresponding config file.
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4_cont00 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4_cont00.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
    ```

By default, if your machine has more than 1 GPU then automatically a unique GPU will be assigned per trainer process. In order to use a specific GPU, you should pass _--gpu_id=*_ command line parameter and must have _GLOBAL_WORLD_SIZE == 1_ evaluate to true in the code.
* Train RND agent in Montezuma's Revenge from scratch using only the GPU _cuda:1_:
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8 --gpu_id=1
    ```

---
* Test a RND agent in Montezuma's Revenge:
    1. set _loadModel = True_ in the corresponding config file.
    2. use _--eval_ command line argument when running the code.
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --eval --num_env_per_process 1 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4_eval --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
    ```

---
- __Distributed Training Architecture__
    * The code relies on _torch.distributed_ package to implement distributed training. It is implemented so that every node is assigned _--nproc-per-node_ many agents (indpendent copies of agent_model on the GPU) which gathers rollouts by interacting with the _--num_env_per_process_ many parallel environment workers (CPU) and trains the agent. These processes have an instance of the gym environment and are used solely to interact with these environments in a parallelized manner. Every agent(trainer) process sends sends actions to the environment worker processes in their node and gathers interactions with the environments. Then, these interactions are used to train the model. Gradients across agent workers are synchronized by making use of the _DistributedDataParallel_ module of _torch_.
    
    * To get a better understanding check out the example below.
    agents_group processes have an instance of RNDAgent and perform optimizations.
    env_workers_group processes have an instance of the environment and are used to interact with the parallelized environments.
        ```txt
        Example:

            Available from torchrun:
                nnodes: number of nodes = 3
                nproc_per_node: number of processes per node = 2
                num_env_per_process = 4
            ---

            ************** NODE 0:
            LOCAL_RANK 0: GPU (cuda:0) --> 4 environment processes
            LOCAL_RANK 1: GPU (cuda:1) --> 4 environment processes
            **************
            ...

            ************** NODE 1:
            LOCAL_RANK 0: GPU (cuda:0) --> 4 environment processes
            LOCAL_RANK 1: GPU (cuda:1) --> 4 environment processes
            ************
            ...

            ************** NODE 2:
            LOCAL_RANK 0: GPU (cuda:0) --> 4 environment processes
            LOCAL_RANK 1: GPU (cuda:1) --> 4 environment processes
            ************

                                -node0-  -node1-   -node2-
                GLOBARL_RANK:     0,1,     2,3,      4,5   
                                   *        *        *
                Note that each rank has a copy of RNDAgent and 4 parallelized environment processes to gather rollouts for training. Their parameters get synchronized after every update to the model parameters.
        ```

---
### Profiling
* Profiling with Scalene (torchrun Example):
    * Logs are directly outputted to browser:
        ```bash
        python -m scalene --- -m torch.distributed.run --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --scalene_profiling 3
        ```
    * Logs are not opened on the browser but are logged to an output file (json):
        ```bash
        python -m scalene --profile-all --cli --json --outfile scaleneProfiler00_modifiedRND00.json --- -m torch.distributed.run --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --scalene_profiling 3
        ```
        or
        ```bash
        make scalene_profiling
        ```
    * Then the logged outputs (json) can be visualized in the browser by running:
    ```bash
    scalene --viewer
    ```
    and inputting the json file (e.g. _scaleneProfiler00_modifiedRND00.json_)

* Profiling with Pytorch Profiler (torchrun Example):
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --pytorch_profiling
    ```
    or
    ```bash
    make pytorch_profiling
    ```
    * To see the profiling results run:
        ```bash
        make start_tensorboard_profiles
        ```
* Profiling with line_profiler:
    ```bash
    LOCAL_RANK=0 RANK=0 LOCAL_WORLD_SIZE=1 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 kernprof -l main.py --train --num_env_per_process 64 --config_path=./configs/exp21/Montezuma/config_modifiedRND_BYOL_2048embed_paperArchitecture_Montezuma.conf --log_name=montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_seed42_exp21-4 --save_model_path=checkpoints/exp21/Montezuma/montezuma_modifiedRND_BYOL_2048embed_paperArchitecture_exp21-4.ckpt --scalene_profiling 10
    ```
    * To see results run:
        ```bash
        python -m line_profiler -rmt "main.py.lprof"
        ```
    Note that we are not using kernprof with _torchrun_ but with _main.py_ directly. This stems from _torchrun_ spawning a new process to run _main.py_ which the _kernprof_ cannot profile due to being a subprocess and not the main process (i.e. _python -m torch.distributed.run_).
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

### Supported OpenAI GYM Environments
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


### Appendix:
__Model Predictive Exploration:__
* Random Network Distillation (RND): 
    
    * Paper: https://arxiv.org/abs/1810.12894
    * Code: https://github.com/jcwleo/random-network-distillation-pytorch

    ---

__Non-Contrastive Representation Learning:__
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
