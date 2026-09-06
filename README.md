# CICEMOK

CICEMOK is a research-oriented Python toolkit for battery-model studies. It
connects COMSOL and PyBaMM simulations, including custom grouped-parameter
Doyle-Fuller-Newman (DFN) models, to uncertainty quantification, sensitivity
analysis, parameter fitting, and experiment-design workflows.

In practical terms, the project helps you:

- run a battery model over a distribution of uncertain parameters;
- fit polynomial-chaos or sparse-regression surrogate models;
- calculate Sobol sensitivity indices for model outputs;
- calibrate parameters against reference curves and explore electrode balance;
- generate and assess current profiles for informative experiments; and
- import and analyse GITT or ICI cycler data.

This is research code, not a polished command-line application. The examples are
valuable workflow templates, but many expect external model files, generated
pickle artifacts, or substantial compute resources.

## Installation and prerequisites

The package metadata targets Python 3.11. From a clean virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The declared dependencies are in `requirements.txt`. Individual workflows have
additional requirements:

- COMSOL workflows require a local licensed COMSOL installation, a compatible
  `mph` setup, and an `.mph` model with the identifiers expected by the script.
- EIS simulations use `pybammeis`, which is imported by `pybammrun.py` but is not
  explicitly declared in `requirements.txt`.
- Some alternative sensitivity implementations also use optional packages such
  as `dill`, `numpoly`, and `TorchSisso`.

## Module guide

| Module | What it is for |
| --- | --- |
| `configuration.py` | Shared dataclasses. `ComsolConfiguration`, `PybammConfiguration`, and `SensitivityConfiguration` hold model, parameter, output, and uncertainty settings; `ExperimentType` selects constant-current, profile, PyBaMM experiment, or EIS modes. |
| `experiments.py` | Loads CSV/XLSX cycler data and provides `GITTExperiment` and `ICIExperiment` helpers to segment constant-current/rest steps and estimate fitted slope/resistance-style quantities. |
| `comsol.py` | Thin `mph` adapter: starts COMSOL, loads a model, applies parameter vectors, current events, timesteps, and initial SOC, then solves it. It includes process-pool worker setup. |
| `pybammrun.py` | PyBaMM execution adapter. It configures DFN, SPM, SPMe, or `GroupedDFNDoubleLayer`, supplies runtime parameters and operating conditions, solves, and interpolates requested outputs onto `xinterp`. |
| `grouped_dfn.py` | A custom grouped-parameter DFN model implemented on top of PyBaMM. |
| `grouped_dfn_dl.py` | The grouped DFN variant with double-layer / potential-difference formulation. This is the custom model wired directly into `pybammrun.py`. |
| `sensitivity.py` | The main sampling and analysis path: evaluates COMSOL/PyBaMM samples in a pool, handles a small number of failed evaluations, fits polynomial-chaos or sparse surrogates, and derives Sobol indices. |
| `sensitivity_serial.py`, `sensitivity_multiprocess.py`, `sensitivity_chaospy.py`, `sensitivity_uqpy.py` | Alternative and experimental sensitivity implementations using different polynomial and regression back ends. Use these when reproducing their corresponding research scripts rather than assuming they share an identical API. |
| `optimization.py` | Parameter-fitting objectives and runners for PyBaMM and COMSOL, using Pymoo-based PSO/NSGA-II and selected Py-BOBYQA workflows. |
| `ode.py` | **Optimal design of experiments**, not an ODE solver. It generates SOC-bounded random C-rate schedules and evaluates their information content with sensitivity/surrogate methods. |
| `balancing.py` | Fits electrode stoichiometry/capacity balance against positive/negative OCP data and full-cell OCV data. |

The package has no curated top-level API yet, so imports normally come from the
specific module, for example `from cicemok.configuration import
PybammConfiguration`.

## Sparse PCE basis adaptation

CICEMOK generates candidate polynomial-chaos-expansion (PCE) bases with a
hyperbolic truncation scheme. The hyperbolic parameter `q` is held in
`SensitivityConfiguration.cross_truncation`: `q = 1` gives a standard
total-degree basis, while smaller values favour lower-order interactions. The
maximum expansion degree is `order`.

The p-q adaptive sparse-PCE routes (`pq-lars-loo`, `pq-omp-loo`, and
`pq-sp-loo`) test candidate bases across the configured expansion-order range and
`q` values. The current implementation scans `p` from `minorder` through `order`
and five `q` values between `0.5` and `cross_truncation`. It uses an analytical
leave-one-out (LOO) cross-validation score computed from the linear PCE
regression to choose the best basis hyperparameters for each output target. The
available sparse regressors are least-angle regression
(LARS), orthogonal matching pursuit (OMP), and subspace pursuit (SP). This
approach follows the adaptive sparse-PCE work of Blatman and Sudret.

Forward-neighbour basis adaptation is also under active development through
`fn-lars-loo`, `fn-omp-loo`, and `fn-sp-loo`. Inspired by Jakeman, Eldred, and
Sargsyan's basis-selection work, it expands a basis through admissible forward
neighbours and uses the LOO score as its selection criterion. This route is not
yet well tested and should currently be treated as experimental.

### References

- G. Blatman and B. Sudret, ["Adaptive sparse polynomial chaos expansion based
  on least angle regression"](https://www.sciencedirect.com/science/article/abs/pii/S0021999110006856),
  *Journal of Computational Physics* 230(6), 2345-2367 (2011),
  doi: [10.1016/j.jcp.2010.12.021](https://doi.org/10.1016/j.jcp.2010.12.021).
- J. D. Jakeman, M. S. Eldred, and K. Sargsyan, ["Enhancing L1-minimization
  estimates of polynomial chaos expansions using basis selection"](https://www.sciencedirect.com/science/article/abs/pii/S0021999115000959),
  *Journal of Computational Physics* 289, 18-34 (2015),
  doi: [10.1016/j.jcp.2015.02.025](https://doi.org/10.1016/j.jcp.2015.02.025).

## Examples

The grouped-parameter DFN examples form a file-based workflow: one script writes
sample/evaluation files, and later scripts read them to fit, validate, and
plot results. Start with small worker and sample counts, and run each script from
its own directory because the files paths are relative.

### Recommended execution order

1. **Generate model evaluations for the training sets.**
   [Reference evaluations](examples/pybamm_grouped_params/reference_evals/get_reference_evals.py)
   runs the grouped-parameter DFN for parameter samples and writes the standardized
   samples, physical samples, and model-response curves used by all PCE methods.

2. **Generate an independent reference/test set.**
   [Reference test-set evaluations](examples/pybamm_grouped_params/reference_evals/get_reference_evals_testset.py)
   produce the larger held-out data set used to measure surrogate generalization
   error. Reduce its initial sample count before a first local run.

3. **Fit one or more PCE families using the same training evaluations.**

   - **Full-order PCE:**
     [fit the PCE](examples/pybamm_grouped_params/pce-regression/get_sensitivity_experiment_many_order.py).
     It writes the basis, Fourier coefficients, and Sobol coefficients. Adjust the
     order/sample lists in the script to compare several full PCE orders.
   - **Sparse LARS, OMP, and SP PCE:** run every response-target chunk in the
     [LARS](examples/pybamm_grouped_params/lars-regression),
     [OMP](examples/pybamm_grouped_params/omp-regression), or
     [SP](examples/pybamm_grouped_params/sp-regression) directory. Then run the
     corresponding `get_combined_data_from_targets.py` script to merge the
     target-wise coefficients into one full response curve. Align the hard-coded
     sample counts and repetition ranges across all target chunks before making a
     complete comparison.

4. **Evaluate errors and compare Sobol indices.**

   - For the full-order PCE, run
     [the generalization-error calculation](examples/pybamm_grouped_params/pce-regression/get_error_test.py),
     then [plot the error distribution](examples/pybamm_grouped_params/pce-regression/plot_error_generalization.py).
     The fitting script already saves first-, total-, and second-order Sobol
     artifacts. [The Sobol bar plot](examples/pybamm_grouped_params/pce-regression/plot_sobol_bar_voltage.py)
     is a useful visualization template. Be sure to change internal hard coded parameters to read the newly generated data.
   - The LARS, OMP, and SP directories each contain matching `get_error_test.py`,
     `plot_error_generalization.py`, and `get_sobol_indices.py` scripts. Run the
     Sobol helper after combining the target-wise fits. There
     is not yet a turnkey grouped-PyBaMM Sobol plotter for LARS, OMP, and SP; use
     the full-PCE plot as a template after making those settings consistent.
     Use the same held-out reference set, sample counts, and response targets
     across all four PCE families for a meaningful comparison.

5. **Use the brute-force Monte Carlo/Sobol route as a direct-model baseline.**
   [The Monte Carlo sensitivity script](examples/pybamm_grouped_params/montecarlo/get_sensitivity_experiment.py)
   evaluates the model directly with SALib Sobol samples instead of fitting a PCE.
   Its saved results can be examined with the
   [bar-chart](examples/pybamm_grouped_params/montecarlo/plot_statistics_bar_voltage.py),
   [heatmap](examples/pybamm_grouped_params/montecarlo/plot_statistics_heatmap_voltage.py),
   and [sample-count](examples/pybamm_grouped_params/montecarlo/plot_statistics_vs_nsamples.py)
   scripts. Treat these plotting scripts as templates too: their parameter labels
   and names must match the direct-model run. The bootstrap scripts in
   the same folder can be used to study sampling variability.

Keeping the model configuration, parameter distribution, response grid, and
random-repetition plan fixed across these branches makes the PCE and direct-model
Sobol results comparable.

## Repository layout and caveats

- `examples/` contains COMSOL and grouped-PyBaMM regression, Monte Carlo, and
  surrogate-analysis workflows. Many scripts deliberately persist intermediate
  `.pkl` artifacts for later plotting or comparison.
- `testing/` contains exploratory research scripts, not a self-contained
  automated test suite; several require local model/data artifacts. Do not assume
  `pytest` will run the repository end-to-end.
- `typings/SALib/` provides local SALib type stubs.
- `docs/agent-specs/` contains the engineering-agent role specifications used to
  maintain this repository.

The COMSOL, optimal-design, and some optimization paths are environment- and
version-sensitive. Treat their scripts as reproducible research starting points:
check parameter names, model identifiers, file paths, sample counts, and current
`sensitivity.py` interfaces before committing a long run.
