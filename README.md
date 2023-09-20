# Assignment 5: Mixed complementarity programming and open-loop dynamic games

In this assignment, you will learn how to use a mixed complementarity problem (MCP) solver and see how to formulate open-loop dynamic games as MCPs. By the end of this assignment, you should:

- have a working familiarity with Julia's symbolic manipulation toolchain, [Symbolics](https://symbolics.juliasymbolics.org/stable/)
- be comfortable formulating and solving nonlinear (but smooth) MCPs using [ParametricMCPs](https://github.com/lassepe/ParametricMCPs.jl)
- get a feeling for when the underlying solver ([PATH](https://pages.cs.wisc.edu/~ferris/path.html)) works or is brittle

As in previous assignments, some starter code is provided and the objective is to pass all unit tests in the `test/` directory. **Unlike other assignments**, these tests will **not** run automatically whenever you push a new commit (in an effort to save on CI credits... the graphics tools take too long to compile). Instead, you must check your implementation locally as described below and post a screenshot to the `Feedback` Pull Request. **Do not modify these tests in your `main` branch. If you feel you must, do so in a separate branch containing _only_ the test modifications, open a Pull Request to `main`, and add me as a reviewer so that I can approve the changes.**

## Setup

As before, this assignment is structured as a Julia package. To activate this assignment's package, type
```console
julia> ]
pkg> activate .
  Activating environment at `<path to repo>/Project.toml`
(Assignment5) pkg>
```
Now exit package mode by hitting the `[delete]` key. You should see the regular Julia REPL prompt. Type:
```console
julia> using Revise
julia> using Assignment5
```
You are now ready to start working on the assignment.

## Part 1: The single-player case

Open the file `src/parametric_optimization_problem.jl`, in which you will encode a "parametric optimization problem" as a MCP. Parametric optimization problems are just optimization problems where the objective and/or constraints depend upon a set of parameters (i.e., rather than only upon the decision variables). For example, if this is a trajectory optimization problem, the parameter could represent a reference speed. The [ParametricMCPs](https://github.com/lassepe/ParametricMCPs.jl) package will let us formulate and solve parametric MCPs using the [PATH](https://pages.cs.wisc.edu/~ferris/path.html) solver, and additionally _differentiate the solution map with respect to the parameter_. We won't be using that functionality in this assignment, but you may find it useful for your projects/research!

Follow the directions in the file to populate the missing pieces. When you run the tests in `test/runtests.jl` (per the directions below) you should be able to pass the `OptimizationTests`.

## Part 2: The multi-player case

Replicate your work from Part 1 and extend it to the multi-player case by filling in the missing sections of `src/parametric_game.jl`. Note here that we include the possibility that players share both equality and inequality constraints. When you run tests, you should pass the `GameTests`.

## Autograde your work

**Unlike other assignments**, your work will **not*** be automatically graded every time you push a commit to GitHub. Instead, you must run tests locally and post a screenshot of the results in the `Feedback` Pull Request. To run tests locally in the REPL you can type:
```console
julia> ]
(Assignment5) pkg> test
```

Alternatively, you can run:
```console
julia> include("test/runtests.jl")
```
once you have activated the `test` environment.

As above: **Do not modify these tests in your `main` branch. If you feel you must, do so in a separate branch containing _only_ your test modifications, open a Pull Request to `main`, and add me as a reviewer so that I can approve the changes.**

## Examples

The `examples/trajectory_game_example.jl` file encodes a trajectory game as a MCP, using your implementation in `src/parametric_game.jl`. You shouldn't need to modify anything to make it run:

```console
julia> includet("examples/trajectory_game_example.jl")
julia> (; sim_steps, game) = main();
```

This will create a video of the animated result called `sim_steps.mp4`. Please upload this file to your `Feedback` Pull Request.

## Bonus for the curious

Play around with the objective (and/or constraints) in this example, and try to generate some interesting behavior. You may (will) find that the solver fails sometimes: try to understand when this happens and why. Put your conclusions (and supporting evidence, e.g. videos, snapshots of videos) in the `Feedback` Pull Request.

## Final note

In the auto-generated `Feedback` Pull Request, please briefly comment on (a) roughly how long this assignment took for you to complete, and (b) any specific suggestions for improving the assignment the next time the course is offered.
