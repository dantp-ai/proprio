# Probabilistic Perception for a Robotic Arm

**Unsupervised classification of sensor readings — *self* vs. *background* vs. *anomaly* — with no explicit geometry or kinematics.**

*Deep Learning & Robotics Challenge 2018 · Team **The Boring Panda** — Daniel Plop & Giorgio Giannone · in collaboration with the [argmax.ai](https://argmax.ai) group.*

---

## The problem

We were given a 7-DOF **Panda** robotic arm with **9 single-point lidars** and an **RGBD camera** on the end-effector, controllable point-to-point via a Python API.

**Goal:** label every incoming sensor reading as **self**, **background**, or **other** — and do it with *unsupervised / semi-supervised learning* instead of hand-built geometric models.

The hypothesis: perception on a manipulator can be solved as a *statistical* problem, without explicitly modelling the robot's configuration or geometry. The payoff is a model that generalizes across environments and **quantifies its own uncertainty**; exactly what is nice to have when the world is dynamic and labels are unavailable.

## Results

| Module | Task | Headline metric |
| --- | --- | --- |
| **Anomaly detection** | Is something new in the scene? | Threshold-free, calibrated uncertainty (predictive σ) |
| **Clustering** | Self vs. background | **89.8%** global accuracy · **69.4%** recall on `self` (3,000 labeled lidar points) |
| **Collision detection** | Anomaly vs. likely collision | **F1 0.61** avg · up to **0.96** per lidar (2,000 points) |

No manual labeling in the loop, no per-sensor heuristics; every decision results from a likelihood.

## Why not just use geometry?

Classical robotics solves *well-defined* tasks beautifully with geometry and control — no learning required. But once the task is loosely specified, the agent isn't perfectly known, and the environment keeps changing, those assumptions break.

|         Geometry-first         |        Learning-first        |
| :----------------------------: | :--------------------------: |
| ![](./project_dlrc2018/images/geometry_based.jpg) | ![](./project_dlrc2018/images/learning_based.jpg) |
| Build an exact system model, add ML for narrow tasks. | Build a statistical model, *inject* geometric/task priors on top. |

Supervised learning is the wrong fit here too: labeling thousands of sensor points per second is impractical and high-variance, discriminative models assume train/test share a distribution (whereas domain shift is what is happens practically in robotics), and they output point-estimates instead of uncertainty. So we for **unsupervised** learning.

## From signal to structure

Logging lidar over a fixed trajectory immediately revealed a clean **multi-modal** structure — the signature of distinct physical regimes (self / background / moving object).

| ![](./project_dlrc2018/images/jointplot_trajectory0.jpg) |
| :---: |
| Per-lidar temporal traces (mm) with density estimates over one trajectory. |

A Gaussian Mixture Model on a tiny hand-crafted feature confirmed the idea. From a short window of a time series $x(t)$, we summarize the temporal-difference transform

$$
f(x_t) = \lvert x(t+1) - x(t) \rvert
$$

with `[max; std]`. On 30 windows (10 points each), the GMM clustered **70%** correctly — and the misses were dynamic samples genuinely indistinguishable from static ones in this embedding. If a 2-D hand-crafted feature gets us this far, a learned representation should solve the full task.

| ![](./project_dlrc2018/images/gt_gmm.jpg) | ![](./project_dlrc2018/images/pred_gmm.jpg) |
| :---: | :---: |
| Ground truth: dynamic (red) vs. static (blue). | GMM clustering prediction. |

## The sensing framework

A hierarchical pipeline of three modules, each answering one question. Input: 9 lidar readings + 7 joint positions (variants also consume depth images).

| ![](./project_dlrc2018/images/framework.jpg) |
| :---: |
| Anomaly → Clustering → Collision: a hierarchy of statistical decisions over raw sensor input. |

**Notation.** $y_{ij}$ is lidar reading $j$ of sample $i$; $y_{i,st}$ a depth pixel; $x_i$ the robot state (joint angles); $\pi_k$ a cluster selector.

### 1. Anomaly detection: *"is anything new?"*

Solve a **proxy regression** task: an MLP (two 256-unit `tanh` layers) predicts, per lidar, a **mean and a standard deviation** from the robot state, trained to minimize negative log-likelihood. When the model can't reconstruct a reading within its own confidence interval, that reading is an anomaly. The threshold is the *learned* predictive uncertainty — no heuristics.

$$
p(Y \mid X) = \prod_i \prod_j \mathcal{N}\!\left(y_{ij}^{\text{lidar}} \mid \mu_j(x_i),\, \sigma_j(x_i)^2\right) \times \prod_{s,t} \mathcal{N}\!\left(y_{i,st}^{\text{depth}} \mid \mu_{st}(x_i),\, \sigma^2(x_i)\right)
$$

$$
\mathcal{L}(Y, X) = -\sum_i \log p(y_i \mid x_i)
$$

| ![](./project_dlrc2018/images/anomaly_blog.jpg) |
| :---: |
| Lidar #3: injected anomalies fall outside the model's predicted range. |

### 2. Clustering: *"self or background?"*

If a reading is *normal*, a **selector network** mimicking a GMM splits it into two learned modes. Same likelihood, now with cluster weights $\pi_{jk}(x_i)$ as the only addition; a 10-point window feeds in temporal context and filters noise.

$$
p(y_{ij}^{\text{lidar}} \mid x_i) = \mathcal{N}\!\left(y_{ij}^{\text{lidar}} \;\middle|\; \sum_k \pi_{jk}(x_i)\,\mu_{jk}(x_i),\; \sum_k \pi_{jk}(x_i)\,\sigma_{jk}(x_i)^2\right)
$$

| ![](./project_dlrc2018/images/cl_blog.jpg) |
| :---: |
| Lidar #3, controlled experiment. Red lines mark the ground-truth bounds of `self`. |

**→ 89.8% global accuracy, 69.4% recall on `self`** (3,000 labeled points).

### 3 · Collision detection — *"new object or impact?"*

Framed as **multi-label binary classification**: corrupt random columns of a 10-point window with Gaussian noise (~50 mm), label originals `0` and perturbations `1`, and learn to separate a generic anomaly from a likely collision — independently per lidar.

Per-lidar results for the collision class (2,000 test points):

| metric | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | **Avg** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 100 | 180 | 140 | 180 | 100 | 170 | 200 | 160 | 180 | 156.6 |
| Sensitivity | 0.96 | 0.33 | 0.22 | 0.56 | 0.84 | 0.97 | 0.22 | 0.20 | 0.57 | **0.54** |
| IoU | 0.92 | 0.32 | 0.20 | 0.50 | 0.73 | 0.91 | 0.19 | 0.17 | 0.50 | **0.49** |
| F1 | 0.96 | 0.49 | 0.33 | 0.67 | 0.84 | 0.95 | 0.32 | 0.29 | 0.66 | **0.61** |

## What worked, what's next

**Worked:** a single statistical pipeline senses the environment with no per-sensor heuristics, learns reusable structure, and reports calibrated uncertainty on every prediction.

**Open:** the unsupervised classifier doesn't yet generalize fully, and the representation isn't directly actionable — perceiving a dynamic scene is not the same as knowing how to act in it.

**Next:** latent-variable models for richer representations and sequence models for temporal dynamics, to make the perception layer robust enough to act on.

## References

- [Variational Inference for On-line Anomaly Detection in High-Dimensional Time Series](https://arxiv.org/abs/1602.07109)
- [Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders](https://openreview.net/forum?id=SJx7Jrtgl)
- [Learning by Association — A Versatile Semi-Supervised Training Method](https://arxiv.org/abs/1706.00909)
- [Generative Ensembles for Robust Anomaly Detection](https://arxiv.org/abs/1810.01392)
