# power-grid-ml

**Differentiable, GPU-ready harmonic power flow and machine learning for power grids** — a
suite of PyTorch packages for harmonic power-quality simulation, harmonic state
estimation, synthetic distribution-grid generation, real-grid datasets, and a dashboard.

The physics is one autograd tape: gradients flow from grid parameters (down to conductor
geometry) through Y-bus assembly and the complex solve `Y(h)·V(h) = I(h)` to every result,
on CPU or CUDA, batched over thousands of scenarios. The same engine is a forward
simulator, a differentiable physics layer for learning, and an inverse / parameter-recovery
tool. Validated against pandapower, OpenDSS and power-grid-model.

## Packages

| repository | install | import | what it is |
|---|---|---|---|
| [pgml](https://github.com/power-grid-ml/pgml) | `pip install power-grid-ml` | `pgml` | the differentiable, GPU-ready harmonic power-flow engine — the base of everything else |
| [pgl](https://github.com/power-grid-ml/pgl) | `pip install power-grid-learn` | `pgl` | harmonic state estimation from few measurements: DNN / GNN / Graphormer models, physics-informed decode, training curriculum, deployment on measured data |
| [pgg](https://github.com/power-grid-ml/pgg) | `pip install power-grid-gen` | `pgg` | quality-diversity generation of feasible low-voltage distribution grids (CVT-MAP-Elites + differentiable repair) |
| [pghub](https://github.com/power-grid-ml/pghub) | `pip install power-grid-hub` | `pghub` | real distribution-grid datasets (ding0, SimBench / Kerber, pandapower reference networks) as `pgml` grids, structural metrics, whole-graph embeddings |
| [pgd](https://github.com/power-grid-ml/pgd) | `pip install power-grid-dash` | `pgd` | a FastAPI service + web UI over simulation, state estimation and live measurements |
| [power-grid-suite](https://github.com/power-grid-ml/power-grid-suite) | `pip install power-grid-suite` | — | everything above in one tested combination; for developers, every package editable in one environment (submodules + pixi) and the SLURM cluster jobs — **start here while nothing is on PyPI** |

Import names stay short (`pgml`, `pgl`, …); the distributions carry the `power-grid-*`
names. Every package depends only on the public API of the packages below it — `pgml`
depends on nothing else in the suite. **Not on PyPI yet** — publication follows the engine
paper; until then install from source as described in the
[power-grid-suite README](https://github.com/power-grid-ml/power-grid-suite#installing--what-works-right-now).

## Documentation, papers, license

- **License**: [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) — use and
  integrate freely, share modifications to the covered files.
- **Contributing**: issues and pull requests are welcome in each package repository; the
  [code of conduct](https://github.com/power-grid-ml/.github/blob/main/CODE_OF_CONDUCT.md)
  applies across the organization.

## AI usage disclaimer
The libraries in this organization were developed with extensive assistance of large language models by Anthropic and OpenAI. Commit messages include co-authorship and references to the model family that was used. Results were validated against existing modeling frameworks (pandapower, OpenDSS, power-grid-model) and key sections of code were reviewed manually. 

Developed at the Cologne Institute for Renewable Energy (CIRE), TH Köln.
