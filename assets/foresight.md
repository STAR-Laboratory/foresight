# Foresight

[[paper](https://arxiv.org/abs/2408.12588)][[blog](https://arxiv.org/abs/2403.10266)]

Pyramid Attention Broadcast(PAB)(#pyramid-attention-broadcastpab)
- [Foresight](#foresight)
  - [Insights](#insights)
  - [Pyramid Attention Broadcast (PAB) Mechanism](#pyramid-attention-broadcast-pab-mechanism)
  - [Experimental Results](#experimental-results)


We introduce Pyramid Attention Broadcast (PAB), the first approach that achieves real-time DiT-based video generation. By mitigating redundant attention computation, PAB achieves up to 21.6 FPS with 10.6x acceleration, without sacrificing quality across popular DiT-based video generation models including Open-Sora, Open-Sora-Plan, and Latte. Notably, as a training-free approach, PAB can enpower any future DiT-based video generation models with real-time capabilities.

## Insights

![method](../assets/figures/pab_motivation.png)

Our study reveals two key insights of three **attention mechanisms** within video diffusion transformers:
- First, attention differences across time steps exhibit a U-shaped pattern, with significant variations occurring during the first and last 15% of steps, while the middle 70% of steps show very stable, minor differences.
- Second, within the stable middle segment, the variability differs among attention types:
    - **Spatial attention** varies the most, involving high-frequency elements like edges and textures;
    - **Temporal attention** exhibits mid-frequency variations related to movements and dynamics in videos;
    - **Cross-modal attention** is the most stable, linking text with video content, analogous to low-frequency signals reflecting textual semantics.

## Pyramid Attention Broadcast (PAB) Mechanism

![method](../assets/figures/pab_method.png)

Building on these insights, we propose a **pyramid attention broadcast(PAB)** mechanism to minimize unnecessary computations and optimize the utility of each attention module, as shown in Figure[xx figure] below.

In the middle segment, we broadcast one step's attention outputs to its subsequent several steps, thereby significantly reducing the computational cost on attention modules.

For more efficient broadcast and minimum influence to effect, we set varied broadcast ranges for different attentions based on their stability and differences.
**The smaller the variation in attention, the broader the potential broadcast range.**


## Experimental Results
Here are the results of our experiments, more results are shown in https://oahzxl.github.io/PAB:

![pab_vis](../assets/figures/pab_vis.png)

