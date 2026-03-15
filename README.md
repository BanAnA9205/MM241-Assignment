# 2D Cutting Stock Optimization: Genetic & Greedy Algorithms

## Team of Authors
This project was developed collaboratively by **Group CC04 - Team 02** for the Mathematical Modeling (CO2011) course at the Ho Chi Minh City University of Technology (HCMUT):
* **Nguyễn Văn An** (Leader) - Genetic Algorithm research and performance analysis.
* **Đào Duy Đạt** - LaTeX formatting and problem formulation.
* **Quách Hoàng Phú** - LaTeX formatting and problem formulation.
* **Lê Đặng Khánh Quỳnh** - Case study research, proofreading, and variations of the problem.
* **Lê Hữu Triều** - Greedy Algorithm research and code optimizing.

---

## Overview
The Cutting Stock Problem (CSP) is a classic NP-hard combinatorial optimization challenge commonly found in manufacturing and logistics. The primary objective is to cut larger stock materials (like metal sheets or wooden planks) into specifically sized smaller pieces to fulfill demand while strictly minimizing waste.

This repository contains our mathematical modeling project, where the 2D CSP is analyzed and solved using our custom heuristic and metaheuristic algorithms.

## Demonstration
![Greedy Algorithm](demo/greedy.gif)

---

## Experiment Setup

Before diving into the results, it is important to understand how the algorithms are tested and evaluated. We use the `gym_cutting_stock` environment to simulate the manufacturing process.

### 1. What is Given? (Environment Synchronization)
To ensure a completely fair comparison, every algorithm is tested across multiple episodes (e.g., Episode 0, Episode 1, etc.). At the start of each episode, the environment is reset using a specific mathematical seed corresponding to the episode number (e.g., `seed=0` for Episode 0). 

This guarantees that in any given episode, **all algorithms are given the exact same starting conditions**:
* The exact same number and dimensions of large stock sheets.
* The exact same list of required product dimensions.
* The exact same target quantities for each product.

### 2. Evaluation Metrics
The environment evaluates the cutting patterns using two primary metrics:
* **Trim Loss:** The ratio of wasted material (empty, unused space) relative to the total area of the stock sheets that were cut into. 
* **Filled Ratio:** The proportion of the total available stock area that was successfully packed with useful pieces. 

### 3. The Objective
Our primary focus is to **minimize Trim Loss**. A lower trim loss means the algorithm successfully arranged the pieces to tightly fit together, thereby wasting the least amount of raw material possible.

---

## Algorithms Implemented

To solve this problem efficiently, we designed and implemented two custom policies:

### 1. Multi-Phase Genetic Algorithm (MPGA)
A highly optimized metaheuristic approach that simulates natural selection to guide the search space toward high-quality cutting patterns. 
* **Guillotine Cut & Placement:** Utilizes the "Difference and Elimination" process to track and update Empty Rectangular Spaces (ERS) dynamically during cuts.
* **Custom Scoring Function:** Evaluates chromosomes not just by trim loss, but by heavily penalizing small wasted spaces and rewarding large, continuous ERS.
* **Multi-Phase Evolution:** Runs multiple evolutionary phases, carrying the top 15% elitist chromosomes into subsequent phases to prevent the algorithm from getting trapped in local optima.

### 2. Custom Greedy Heuristic
A step-by-step deterministic decision-making algorithm that provides rapid, near-optimal solutions.
* **Dimensional Reorientation:** Automatically rotates all stocks and demands so that widths are strictly greater than heights, sorting them by descending width.
* **Two-Phase Packing:** Places pieces "flat down" (parallel to the width) in Phase 1, and "standing up" (perpendicular) in Phase 2 to tightly pack the right and top borders of the stocks.

*Note: You can read our full mathematical formulation, including the Integer Linear Programming (ILP) models and time complexity analysis, in the [`report/`](report/) directory.*

---

## Performance Results

We evaluated our custom algorithms against the provided baseline policies. The data below shows the results of the first 5 strictly synchronized episodes.

### Chart 1: Overall Average Performance
This chart displays the average trim loss across the tested episodes. **Lower is better.**

| Policy | Average Trim Loss | Efficiency Analysis |
| :--- | :--- | :--- |
| **Random Policy** | **~77.7%** | Highly inefficient; random placement leaves massive unusable gaps. |
| **Baseline Greedy** | **~27.0%** | Standard greedy logic; adequately fills space but lacks spatial optimization. |
| **Our Custom Greedy** | **~18.4%** | Highly efficient for real-time applications; spatial sorting and dual-phase packing aggressively reduces waste corners. |
| **Our Genetic Algorithm** | **~12.8%** | **Best Performance;** evolutionary backtracking and ERS optimization consistently find near-global optima. |

### Chart 2: Detailed Performance per Episode
This breakdown proves that our custom algorithms consistently outperform the baselines across various randomly generated scenarios (seeds). **Lower Trim Loss is better.**

| Episode | Random Policy | Baseline Greedy | Our Custom Greedy | Our Genetic Algorithm |
| :---: | :---: | :---: | :---: | :---: |
| **Ep 0** | 0.686 | 0.450 | 0.266 | **0.148** |
| **Ep 1** | 0.864 | 0.209 | 0.190 | **0.129** |
| **Ep 2** | 0.773 | 0.235 | 0.222 | **0.137** |
| **Ep 3** | 0.866 | 0.265 | 0.131 | **0.091** |
| **Ep 4** | 0.697 | 0.190 | **0.110** | 0.133 |

*(Note: While Filled Ratios vary depending on how many stocks each algorithm decides to pull from to satisfy the demand, the Trim Loss proves that our algorithms pack the pieces much more densely within the stocks they do use.)*

--- 

## Installation

To run this project, ensure you have Python installed, then clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
You can observe the agents solving the environment in real-time by executing the main script:

```bash
python main.py
```

## License
This project is open-source and distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.

