# Setting up Julia 

Table of Contents
- [Setting up Julia](#setting-up-julia)
  - [Installation](#installation)
  - [Utilities](#utilities)
    - [Version Control](#version-control)
    - [Debugging](#debugging)
  - [Projects](#projects)
    - [Setting up a new project](#setting-up-a-new-project)
    - [Working on an existing project](#working-on-an-existing-project)

## Installation
Navigate to [https://julialang.org](https://julialang.org) and follow the instructions to download and install the latest version of the Julia language.

## Utilities 

### Version Control 

- [juliaup](https://github.com/JuliaLang/juliaup)
- [jill.py](https://github.com/johnnychen94/jill.py)

### Debugging 

- [Revise.jl](https://github.com/timholy/Revise.jl) Modify code and use the changes without restarting Julia.*
- [Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl) Set breakpoints.*
- [Cthulhu.jl](https://github.com/JuliaDebug/Cthulhu.jl) Type inference issues.
- [JET.jl](https://github.com/aviatesk/JET.jl) Type inference

*These packages are so useful, it's a good idea to have them installed globally so that they're available in any project.

## Projects 

### Setting up a new project

Useful guide by Martin Maas: [Creating a New Project
](https://www.matecdev.com/posts/julia-create-project.html)

### Working on an existing project

1. Open a new terminal window and navigate to your repository.
2. At the command prompt, type:
```console
$ julia
```
3. This will open the Julia REPL. If this is your first time in Julia, make sure you have the `Revise` package installed globally. It will enable you to make code changes in a text editor and recompile code dynamically so that you don't need to restart the REPL every time you make a code change. To install `Revise` globally, enter
```console
julia> import Pkg; Pkg.add("Revise")
```
4. Now, you're ready to start working on your project. From (1) your terminal (and the REPL) should already be working in the root directory of this repository. If not, `cd` there (note that Julia has both `cd()` and `pwd()` functions that work just like their `bash` counterparts). Enter package mode in the REPL as follows:
```console
julia> ]
```
1. If the repository is setup as a package, activate it by typing
```console
pkg> activate .
  Activating environment at `<path to repo>/Project.toml`
(My_Repo) pkg>
```
1. Now exit package mode by hitting the `[delete]` key. You should see the regular Julia REPL prompt. Type:
```console
julia> using Revise
julia> using My_Repo
```
1. Congrats! You're all ready to start working on the project.