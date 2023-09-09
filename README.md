# Representation Learning for Model-Predictive Exploration in Reinforcement Learning

## Motivation:
It is hard for Reinforcement Learning Agents to learn with high dimensional state spaces. This is why it is important to have smaller state representations. Adding auxiliary losses for representation learning is one way to deal with this. 
Furthermore, exploration is crucial for Reinforcement Learning problems, especially for problems with high dimensional state spaces. There are many approaches which were proposed to deal with this issue. One of these methods is based on "prediction-error". With this method, one predicts "something" (e.g. next state, reward) then we compare it against our actual observations. If the discrepancy between those two is high, one concludes that further exploration of such states is required to decrease the error. It gets harder to perform this with high dimensional spaces for reasons such as curse of dimensionality, and the noise present. 
Following from these two findings, we believe that for "prediction-based" exploration methods, receiving support from representation learning methods appears as a viable solution.

---

### Installation:
* Installation with Docker:
    ```bash
    ... # create Image
    ... # create and run Container
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
* Random Network Distillation (RND): 
    
    * Paper: https://arxiv.org/abs/1810.12894
    * Code: https://github.com/jcwleo/random-network-distillation-pytorch