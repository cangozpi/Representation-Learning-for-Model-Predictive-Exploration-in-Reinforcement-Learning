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
### To Run:
* Run RND training code:
    ```bash
    make train
    ```
* Kill RND code (and its subprocesses):
    ```bash
    make kill
    ```


---

### Appendix:
__Model Predictive Exploration:__
* Random Network Distillation (RND): 
    
    * Paper: https://arxiv.org/abs/1810.12894
    * Code: https://github.com/jcwleo/random-network-distillation-pytorch

__Non-Contrastive Representation Learning:__
* BYOL-Explore: Exploration by Bootstrapped Prediction:

    * Paper: https://arxiv.org/abs/2206.08332


* Bootstrap Your Own Latent A new Approach to Self-Supervised Learning (BYOL):

    * Paper: https://arxiv.org/pdf/2006.07733.pdf
    * Code: 
        
        1. https://github.com/The-AI-Summer/byol-cifar10/blob/main/ai_summer_byol_in_cifar10.py#L92
        2. https://github.com/SaeedShurrab/Simple-BYOL/blob/master/byol.py
        3. https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py