# Radial Autonomous Rendezvous Transformer
*Testing the effectiveness of temporal radial attention within SLAB+ASL's Autonomous Rendevous Transformer (ART) and Sequential Convex Programming (SCP) pipeline.*

---

## 1  What is Radial ART?

ART is a Decision-Transformer that learns offline to map orbital states (rtn or roe), return-to-go (fuel budget), and constraint-to-go (keep-out-zone violation budget) to the next $`\Delta v`$ control actions for proximity-operations and docking. Standard ART uses dense GPT-2 attention. Thus, memory and compute scale quadratically with the N=4·K timestep token context. Applying Radial Attention replaces the dense pattern with a log-polar, block-sparse mask of:

1. dense band of width $`w_0`$ around the diagonal 
2. attention span halves every log-distance group 
        
As a result, this requires only $`\mathcal{O} (N \log N )`$ tokens. This procedure should both speed-up inference and require less VRAM with little to no loss in control performance. To integrate the adjustment, LoRA is used to fine-tune and adapt the sparse model to the original dataset.

In a more physically intuitive sense, radial attention allows the transformer to learn by attributing less attention to far off dynamics (far-future or far-past) while maintaining full focus on the near-time dynamics. Given the local nature of classical dynamics, this approach seems somewhat natural.

---

## 2  Implementation Details

1. Load dense base model weights
2. Swap dense blocks for sparse radial blocks
3. Configure LoRA adapters
4. Fine tune on A100 with accelerate+flash-attn (requires fp16)
5. Evaluate with new model in ART-SCP Pipeline

---

## 3  Persisting Issues/TODO:

1. Implement planned features
2. Migrate from colab script
3. Investigate switch back to fp32 (flash-attn limited)

---

## 4  Repository Contains:

This repo contains the src folder with the radial_swap function to hot-swap the dense GPT-2 attention blocks and the modified training and evaluation scripts to fine-tune and evaluate the Radial ART model. The repo also contains the full Radial ART pipeline within a rough colab script.

---

## 5  Testing Results:

Cost and runtime results are shown below.

| Model          |   Cost   |  Runtime |
|----------------|:--------:|:--------:|
|       CVX      |  0.2010  |  0.2083  |
|     CVX+SCP    |  0.2114  |  2.2871  |
|       ART      |  0.2724  |  0.9233  |
|   Radial ART   |  0.1714  |  0.9489  |
|     ART+SCP    |  0.2010  |    -     |
| Radial ART+SCP |  0.2102  |    -     |

*Note: In the following figures ART and ART-SCP are the evaluated Radial ART and Radial ART-SCP models.

<table>
  <tr>
    <td align="center">
      <img src="radial_art_results/figures/pos_3d_split.png" width="400"><br>
      <em>3-D Trajectory With+Without Radial ART+SCP</em>
    </td>
    <td align="center">
      <img src="radial_art_results/figures/roe_vs_time.png" width="400"><br>
      <em>ROE vs. Time</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="radial_art_results/figures/delta_v.png" width="400"><br>
      <em>Delta v Profile</em>
    </td>
    <td align="center">
      <img src="radial_art_results/figures/koz_constr_v2.png" width="400"><br>
      <em>KOZ Constraint</em>
    </td>
  </tr>
</table>

**Positive Results:**
- Memory is indeed reduced with sparse-radial attention masks cutting nearly 70% of attention-matrix memory (and associated large reduction in VRAM). 
- In addition, Radial ART+SCP performs roughly 4.3% worse than ART+SCP, which is not terrible given the reduced memory.

**Negative Results:**
- While Radial ART alone appears to excel, this is the result of violating feasibility/koz and thus cannot be taken in actuality. 
- This conclusion for Radial ART alone is further suggested as the Radial ART+SCP performs slightly worse than ART+SCP (does not take a phantom 0.1714 cost solution).
- Most notably, Radial ART runtime is roughly even with that of ART. This is likely because the small sequnce length and small state space do not yield a large advantage when using sparse representation.
 
 ---
 
## 6  Conclusion
As a result, it seems radial attention does not excel within the ART-SCP pipeline. While reducing some memory, radial attention does not really provide much speed-up within the current use case. However, should longer time horizons be desired in the future, radial attention could prove a valuable tool.

---

## 7  License

This project is licensed under the MIT License – see `LICENSE` for details.

