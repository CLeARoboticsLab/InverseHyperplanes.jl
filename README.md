# InverseHyperplanes.jl

Collision-free trajectories for non-cooperative multi-agent systems using rotating hyperplanes constraints learned from expert data. 

An example using the Hill-Clohessy-Wiltshire equations for relative orbital motion:
<table>
  <tr>
    <td style="height: 10px;">1. Noisy expert data</td>
    <td style="height: 10px;">2. Inferred hyperplanes</td>
    <td style="height: 10px;">3. Collision-free trajectory</td>
  </tr>
  <tr>
    <td valign="top"><img src="figure/pull_expert.gif"  height="250"></td>
    <td valign="top"><img src="figure/pull_inverse.gif" height="250"></td>
    <td valign="top"><img src="figure/pull_3D.gif"      height="250"></td>
  </tr>
 </table>

> Note: For any questions on how to use it please reach out to me at [fernandopalafox@utexas.edu](mailto:fernandopalafox@utexa.edu) or open an issue.

## Paper Abstract 

A core challenge of multi-robot interactions is collision avoidance among robots with potentially conflicting objectives. We propose a game-theoretic method for collision avoidance based on rotating hyperplane constraints. These constraints ensure collision avoidance by defining separating hyperplanes that rotate around a keep-out zone centered on certain robots. Since it is challenging to select the parameters that define a hyperplane without introducing infeasibilities, we propose to learn them from an expert trajectory i.e., one collected by recording human operators. To do so, we solve for the parameters whose corresponding equilibrium trajectory best matches the expert trajectory.

Read the full paper [here](https://arxiv.org/abs/2311.09439).

## Setup

After cloning the repository, you can install all dependencies at the
versions recorded in the `Manifest.toml` by running: 

1. Navigate to the directory where you cloned this repo.
2. Start Julia in project mode: `julia --project`.
3. Hit `]` to enter package mode and run: `pkg> instantiate`.

## Directory Layout

- `src/` contains the implementations of our method and the baseline for
  solving the inverse game from expert data. Beyond that it contains implementations of forward game
  solvers and visualization utilities.

- `experiments/` contains the code for reproducing the examples shown in the [paper](https://arxiv.org/abs/2311.09439). 

## Running your own experiments
This package uses the proprietary PATH solver under the hood (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)).
Therefore, you will need a license key to solve larger problems.
However, by courtesy of Steven Dirkse, Michael Ferris, and Tudd Munson,
[temporary licenses are available free of charge](https://pages.cs.wisc.edu/~ferris/path.html).
Please consult the documentation of [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl) to learn about loading the license key.