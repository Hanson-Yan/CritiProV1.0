# CritiPro V1.0

CritiPro is a critical‑node‑aware network topology obfuscation framework.  
It protects critical infrastructure nodes against topology inference attacks while keeping deployment cost controllable and preserving traffic performance.

The system is composed of three main modules:

1. **Critical node identification module**
2. **Topology obfuscation module (CritiPro)**
3. **Deployment optimization module**

CritiPro is evaluated against two baselines:

- **ProTO** – a cost‑aware topology obfuscation method based on delay manipulation
- **AntiTomo** – an obfuscation method originally operating on adjacency / routing matrices

The repository also includes:

- **MininetTop** – Mininet + Ryu based network emulation and probing/inference
- **Experiment** – scripts for quantitative evaluation, ablation studies, and visualization

---

## Repository Structure

At a high level:

- [`critinode_model/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/critinode_model)
  - Critical node and link identification algorithms and metrics.
- [`topo_obfuscation_ccs/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs)
  - CritiPro’s core topology obfuscation algorithm (multi‑objective optimization, NSGA‑II, hill‑climbing, similarity constraints).
- [`topo_deployment/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment)
  - Deployment optimization solvers and utilities (operation matrices, delay solvers, etc.).
- [`ProTO/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO)
  - Implementation of the ProTO baseline (topology obfuscation + deployment optimization).
- [`AntiTomo/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo)
  - Implementation of the AntiTomo baseline and its integration with ProTO‑style deployment optimization.
- [`MininetTop/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop)
  - Mininet/Ryu‑based probing, inference, and emulation of original and obfuscated topologies.
- [`Experiment/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment)
  - Experiment scripts for similarity verification, deployment cost, throughput impact, ablation, and critical node transfer.
- [`README.md`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/README.md)
  - (This file.)

Below we summarize the key components.

---

## 1. Critical Node Identification Module

**Directory:** [`critinode_model/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/critinode_model)

This module identifies structurally critical nodes (and links) from network metrics.

Key files:

- [`critical_node_search.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/critinode_model/critical_node_search.py)  
  - Provides functions such as `identify_key_nodes` and `identify_key_nodes_adaptive`, used by Mininet and the obfuscation module to locate critical nodes.
- [`node_metrics.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/critinode_model/node_metrics.py)  
  - Defines node‑level metrics (degree‑like measures, centrality proxies, etc.) used to score node importance.
- [`link_metrics.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/critinode_model/link_metrics.py)  
  - Defines metrics for link importance.
- [`critical_node_only.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/critinode_model/critical_node_only.py) and [`critical_link_only.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/critinode_model/critical_link_only.py)  
  - Standalone scripts to generate lists of critical nodes or links from pre‑computed metrics files.

Typical data flow:

1. Generate node/link metrics for a given topology.
2. Use `identify_key_nodes` or related functions to obtain the critical node set.
3. Feed the set of critical nodes into the topology obfuscation module to ensure they are protected.

---

## 2. Topology Obfuscation Module (CritiPro)

**Directory:** [`topo_obfuscation_ccs/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs)

This directory implements CritiPro’s core obfuscation algorithm, which simultaneously:

- Minimizes **criticality exposure** of key nodes.
- Minimizes **obfuscation cost** (e.g., required delay manipulations).
- Enforces **structural similarity** constraints so that the obfuscated topology remains close to the original one.

Key ideas (from [`readme.md`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/readme.md) and [`objective.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/objective.py)):

- **Similarity measure**: Uses a structure‑aware measure (network portrait divergence) to capture topological similarity, not just path‑wise metrics.
- **Optimization objectives**:
  - Minimize criticality of visible nodes in the obfuscated topology.
  - Minimize deployment cost / perturbation magnitude.
- **Hard constraints**:
  - Critical elements must be protected.
  - Topology must remain connected.
- **Soft constraints**:
  - Structural similarity maintained within a target range (e.g., between 0.6 and 0.9).
  - Physical deployability (e.g., virtual delays must remain larger than physical delays).
- **Perturbation locality**:
  - Obfuscation is restricted within a limited number of hops (`b`‑hop neighborhood) to keep changes deployable and tightly coupled with structural similarity.

Important files:

- [`main.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/main.py)  
  - End‑to‑end pipeline:
    - Loads adjacency matrices and metrics from `data/`.
    - Encodes sparse edges with [`encoder.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/encoder.py).
    - Evaluates multi‑objective obfuscation using `ObfuscationObjective` from [`objective.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/objective.py).
    - Runs NSGA‑II via [`nsga2_solver.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/nsga2_solver.py), optionally followed by local hill‑climbing (`hill_climb.py`).
    - Uses similarity, metric, and post‑processing utilities from [`utils/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/utils).
    - Produces Pareto‑optimal obfuscated topologies and saves them into `data/topo_X_output_file/`.
- [`objective.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/objective.py)  
  - Defines the joint objective (“min criticality + min cost” with a soft similarity constraint).
- [`nsga2_solver.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/nsga2_solver.py) and [`hill_climb.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/hill_climb.py)  
  - Provide evolutionary and local search components for exploring the solution space.
- [`utils/critical_node.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/utils/critical_node.py), [`utils/metrics.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/utils/metrics.py), [`utils/similarity.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/utils/similarity.py), [`utils/post_process.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/utils/post_process.py)  
  - Support code for metric evaluation, similarity computation, caching, and filtering/selecting final solutions.
- [`draw_topo.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/draw_topo.py)  
  - Visualization of original vs obfuscated topologies and critical nodes.

---

## 3. Deployment Optimization Module

**Directory:** [`topo_deployment/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment)

This module translates obfuscated routing/adjacency structures into deployable delay operations under constraint.

Key components:

- [`data/input_file/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/data/input_file)
  - Example matrices `F.txt`, `M.txt`, `r.txt` used in deployment tests.
- [`operation_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/operation_matrix.py), [`vector_to_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/vector_to_matrix.py), [`matrix_to_vector.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/matrix_to_vector.py)  
  - Conversion between matrix and vector forms of routing/operation matrices.
- [`node_delay_solver.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/node_delay_solver.py), [`operation_solver.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/operation_solver.py)  
  - Core solvers for deriving feasible deployment strategies based on delay constraints.
- [`adam_operation_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/adam_operation_matrix.py), [`pgd_operation_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/pgd_operation_matrix.py), [`torch_operation_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_deployment/torch_operation_matrix.py)  
  - Variants of optimization strategies for optimizing operation matrices.

In the full pipeline, the deployment module:

1. Takes an obfuscated topology (or routing matrix) as input.
2. Solves for an operation/delay vector ensuring constraints such as maximum delay deviation.
3. Outputs per‑link or per‑node delay configurations used in Mininet/Ryu simulations.

---

## 4. Baseline Methods

### 4.1 ProTO

**Directory:** [`ProTO/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO)

ProTO is a cost‑aware topology obfuscation and deployment optimization method used as a baseline for CritiPro.

- [`main.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO/main.py)  
  - Entry point for:
    - Preparing data (converting adjacency matrices to routing matrices, generating obfuscated topologies).
    - Solving the deployment optimization problem.
- [`adjacency_to_routing.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO/adjacency_to_routing.py), [`matrix_to_vector.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO/matrix_to_vector.py), [`generate_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO/generate_matrix.py)  
  - Support utilities for matrix generation and transformation.
- [`draw_topo.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO/draw_topo.py)  
  - Visualization for ProTO’s topologies.
- [`input_file/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO/input_file) and `topo_*_result/`  
  - Example input and resulting topologies.

The local [`readme.md`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/ProTO/readme.md) (Chinese) briefly explains:

- `main.py` as the main entry point.
- Data preparation step (random obfuscation + adjacency‑to‑routing conversion).
- The optimization step for deployment.

### 4.2 AntiTomo

**Directory:** [`AntiTomo/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo)

AntiTomo is another baseline that works directly with adjacency and routing matrices to generate obfuscated topologies. Deployment optimization is reused from ProTO.

- [`main.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo/main.py)  
  - Implements:
    - Data preparation from original topologies.
    - Obfuscation via `AntiTomoDefender` (see code).
    - Deployment solving using ProTO’s formulation (`deploy_solve` function).
- [`adjacency_to_routing.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo/adjacency_to_routing.py), [`generate_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo/generate_matrix.py), [`confuse_matrix.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo/confuse_matrix.py), [`matrix_to_vector.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo/matrix_to_vector.py)  
  - Supporting utilities.

The local [`readme.md`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/AntiTomo/readme.md) notes that:

- AntiTomo operates on adjacency matrices and generates obfuscated routing matrices.
- Deployment is *not* natively included and is borrowed from ProTO.
- The original implementation uses a Python 3.6 environment for PuLP.

---

## 5. Mininet‑Based Simulation and Topology Inference

**Directory:** [`MininetTop/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop)

This module uses **Mininet** and **Ryu** to:

- Emulate original and obfuscated network topologies.
- Generate probing traffic to infer topologies.
- Evaluate throughput and delay impacts before and after deployment.

Key components:

- [`small_topo_test.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/small_topo_test.py)  
  - Main Mininet script:
    - Builds topologies from adjacency matrices in [`topo_matrix/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/topo_matrix).
    - Interacts with a Ryu controller (`my_learning_switch.py` pre‑deployment, `my_delay_switch.py` post‑deployment).
    - Integrates with `critinode_model` to identify critical nodes on live topologies.
    - Supports link flooding tests (`measure_link_flood`), throughput measurements (`measure_throughput`), and various visualization utilities.
- [`my_learning_switch.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/my_learning_switch.py)  
  - Ryu controller for normal learning switch behavior (pre‑deployment).
- [`my_delay_switch.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/my_delay_switch.py)  
  - Ryu controller that introduces additional delays based on deployment vectors.
- [`get_topo/topo_generator.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/get_topo/topo_generator.py), [`probe_simulation/topo_to_tree.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/probe_simulation/topo_to_tree.py)  
  - Scripts to generate and convert topologies into tree forms used for inference.
- [`probe_simulation/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/probe_simulation) and [`probeCode/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/probeCode)  
  - Probing, inference, and visualization utilities (`probe_simulation.py`, `topo_infer_rnj.py`, `topo_infer_plot.py`, etc.).

The local [`readme.md`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/MininetTop/readme.md) (Chinese) describes the typical workflow:

1. Start **two terminals**:
   - One in a Ryu environment (`ryu-env`) to run the controller:
     ```bash
     ryu-manager --observe-links my_learning_switch.py
     ```
   - One base terminal to run Mininet:
     ```bash
     sudo mn -c
     sudo python small_topo_test.py   # path may need adjustment to your Python
     ```
2. Ensure `pingall` succeeds before starting probing.
3. Open xterm windows for hosts (e.g., `xterm h1`) and run probing traffic.
4. Use `topo_generator.py` and `topo_to_tree.py` to generate and convert topologies for inference.

---

## 6. Experiment Module

**Directory:** [`Experiment/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment)

This directory contains scripts for all evaluation procedures used in the project.

Major functions:

- **Data preparation and copying**
  - [`prepare_data.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/prepare_data.py)  
    - Collects outputs from CritiPro, ProTO, and AntiTomo:
      - Confused topology matrices
      - Deployment vectors
    - Organizes them under `topo_X_result/` for experiments.
- **Critical node experiments**
  - [`key_node_static.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/key_node_static.py)  
    - Compares critical node sets between the original and obfuscated topologies of each method (CritiPro, ProTO, AntiTomo).
  - [`compare_critinode.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/compare_critinode.py)  
    - Visualizes original vs obfuscated topologies and highlights critical nodes.
- **Similarity and cost evaluation**
  - [`compute_similarity.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/compute_similarity.py)  
    - Computes topological similarity metrics.
  - [`compute_deploy_cost.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/compute_deploy_cost.py)  
    - Computes deployment cost for each method.
  - [`plot_cost.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/plot_cost.py), [`plot_cost_1x4.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/plot_cost_1x4.py)  
    - Plot deployment cost vs detection frequency for multiple topologies.
  - [`plot_impact.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/plot_impact.py), [`plot_impact_1x4.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/plot_impact_1x4.py)  
    - Plot throughput/impact metrics before and after deployment.
- **Ablation and robustness**
  - [`ablation/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/ablation)  
    - Scripts for ablation studies, randomized experiments, and failure checks.
- **Flooding and Mininet‑based tests**
  - [`flood_test/`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/flood_test)  
    - Scripts and results for link flooding experiments on multiple topologies.

The local [`readme.md`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/readme.md) mentions:

- `test.py` backs up result data into `topo_num_result/` directories.
- The topologies `topo_1`, `topo_2`, `topo_4`, `topo_3` correspond to increasing scales (small → medium → large).
- Similarity‑threshold experiments: random obfuscated topologies are generated under different similarity intervals and compared.
- Mininet deployment impact experiments:
  - Measure throughput before deployment.
  - Deploy delay operations (e.g., 10% of packets delayed over a total of 10,000 costed delays).
  - Measure throughput after deployment using different controllers (`my_learning_switch` vs `my_delay_switch`).

---

## 7. End‑to‑End Workflow (High‑Level)

An example end‑to‑end workflow is:

1. **Prepare original topologies**
   - Use `MininetTop/topo_matrix/topo_X.txt` (or generate new ones via `get_topo/topo_generator.py`).
2. **Identify critical nodes**
   - Run metric calculation and critical node identification using scripts in `critinode_model/`.
3. **Run CritiPro obfuscation**
   - Use [`topo_obfuscation_ccs/main.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/topo_obfuscation_ccs/main.py) to generate Pareto‑optimal obfuscated topologies under similarity and cost constraints.
4. **Solve deployment**
   - Use `topo_deployment/` or ProTO‑style solvers to derive deployable delay operations.
5. **Evaluate baselines**
   - Run corresponding scripts in `ProTO/` and `AntiTomo/` to generate their obfuscated topologies and deployment vectors.
6. **Prepare experiment data**
   - Collect all outputs via [`Experiment/prepare_data.py`](file:///d:/D_download/ChromeDL/CritiProV1.0-main/Experiment/prepare_data.py).
7. **Run experiments and plot results**
   - Run cost, similarity, impact, critical node transfer, and ablation scripts in `Experiment/`.
8. **Mininet validation**
   - Use `MininetTop/` + Ryu to emulate original and obfuscated topologies.
   - Deploy delay operations and measure throughput and flooding resilience.

---

## 8. Requirements and Environment

This project relies on:

- **Python** (multiple versions may be used; some baselines, e.g., AntiTomo + PuLP, were originally run under Python 3.6).
- **Scientific stack**:
  - `numpy`, `scipy`, `matplotlib`
  - `networkx`
- **Optimization / ML libraries**:
  - PuLP or other LP solvers (for ProTO / AntiTomo deployment)
  - PyTorch (for some deployment optimization variants in `topo_deployment/`)
- **Network emulation tools**:
  - **Mininet**
  - **Ryu** SDN controller

Because paths in the scripts are currently hard‑coded to specific directories (e.g., `/home/retr0/Project/TopologyObfu/...`), you will likely need to:

- Adjust file paths to match your local environment.
- Configure virtual environments for:
  - Mininet + Ryu
  - Python 3.x with required dependencies
  - (Optional) a separate Python 3.6 environment for PuLP if reproducing AntiTomo exactly.

---

## 9. How to Get Started

The exact steps depend on your environment, but a typical sequence is:

1. **Clone the repository**
   ```bash
   git clone <this-repo-url>
   cd CritiProV1.0-main
   ```

2. **Set up Python environment(s)**
   - Create a virtual environment and install dependencies (NumPy, SciPy, NetworkX, Matplotlib, etc.).
   - Optionally, create a Python 3.6 environment for PuLP if you want to reproduce AntiTomo’s original setup.

3. **Run CritiPro obfuscation on a sample topology**
   - Adjust hard‑coded paths in `topo_obfuscation_ccs/main.py` to point to your data.
   - Run:
     ```bash
     python topo_obfuscation_ccs/main.py
     ```
   - Inspect outputs under `topo_obfuscation_ccs/data/topo_X_output_file/`.

4. **Run ProTO / AntiTomo baselines**
   - Adjust paths in `ProTO/main.py` and `AntiTomo/main.py`.
   - Run:
     ```bash
     cd ProTO
     python main.py
     cd ../AntiTomo
     python main.py
     ```

5. **Prepare and run experiments**
   - Use `Experiment/prepare_data.py` to gather outputs.
   - Run plotting and analysis scripts in `Experiment/` to reproduce key figures (deployment cost, impact, similarity, critical node transfer).

6. **Mininet validation**
   - Start a Ryu controller:
     ```bash
     ryu-manager --observe-links my_learning_switch.py
     ```
   - Start Mininet with `small_topo_test.py` and follow the probing instructions in `MininetTop/readme.md`.

---

## 10. License and Citation

If you use CritiPro or any part of this repository in your research, please cite the corresponding paper (add the citation entry here).

---
