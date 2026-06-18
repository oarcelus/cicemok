import numpy as np
import pybamm
from pybamm import (
    Event,
    FunctionParameter,
    Parameter,
    ParameterValues,
    Scalar,
    SpatialVariable,
    Variable,
)
from pybamm import t as pybamm_t
from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
    get_min_max_stoichiometries,
)


class GroupedDFN(pybamm.lithium_ion.BaseModel):
    """
    A grouped parameter version of the Doyle-Fuller-Newmann model (DFN).

    Parameters
    ----------
    name : str, optional
        The name of the model.
    **model_kwargs : optional
        Valid PyBaMM model option keys and their values, for example:
        options : dict, optional
            A dictionary of options to customise the behaviour of the PyBaMM model.
        build : bool, optional
            If True, the model is built upon creation (default: False).
    """

    def __init__(self, name="Grouped Doyle-Fuller-Newmann model", **model_kwargs):
        super().__init__(name=name, **model_kwargs)

        pybamm.citations.register("Chen2020")  # for the OCPs
        pybamm.citations.register(
            """
            @article{Hallemans2025,
            title     = {{Physics-Based Battery Model Parametrisation from Impedance Data}},
            author    = {Hallemans, Noël and Courtier, Nicola E. and Please, Colin P. and Planden, Brady and Dhoot, Rishit and Timms, Robert and Chapman, S. Jon and Howey, David and Duncan, Stephen R.},
            journal   = {Journal of the Electrochemical Society},
            volume    = {172},
            number    = {6},
            pages     = {060507},
            year      = {2025},
            publisher = {The Electrochemical Society},
            doi       = {10.1149/1945-7111/add41b}
            }
        """
        )

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = Variable("Discharge capacity [A.h]")
        Qt = Variable("Throughput capacity [A.h]")

        # Particle variables
        sto_s_n = Variable(
            "Negative particle stoichiometry",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative electrode",
            },
            scale=Scalar(1)
        )
        sto_s_p = Variable(
            "Positive particle stoichiometry",
            domain="positive particle",
            auxiliary_domains={
                "secondary": "positive electrode",
            },
            scale=Scalar(1)
        )
        phi_s_n = Variable(
            "Negative electrode potential [V]",
            domain="negative electrode",
        )
        phi_s_p = Variable(
            "Positive electrode potential [V]",
            domain="positive electrode",
        )
        sto_s_n_surf = pybamm.surf(sto_s_n)
        sto_s_p_surf = pybamm.surf(sto_s_p)

        soc_init = Parameter("Initial SoC")
        sto_s_n_0 = Parameter("Minimum negative stoichiometry")
        sto_s_n_100 = Parameter("Maximum negative stoichiometry")
        sto_s_p_100 = Parameter("Minimum positive stoichiometry")
        sto_s_p_0 = Parameter("Maximum positive stoichiometry")
        U_n = self.U(sto_s_n_surf, "negative")
        U_p = self.U(sto_s_p_surf, "positive")
        sto_s_p_init = sto_s_p_0 + (sto_s_p_100 - sto_s_p_0) * soc_init
        sto_s_n_init = sto_s_n_0 + (sto_s_n_100 - sto_s_n_0) * soc_init
        U_n_init = self.U(sto_s_n_init, "negative")
        U_p_init = self.U(sto_s_p_init, "positive")

        # Electrolyte variables
        sto_e_n = Variable(
            "Negative electrolyte stoichiometry",
            domain="negative electrode",
            scale=Scalar(1)
        )
        sto_e_sep = Variable(
            "Separator electrolyte stoichiometry",
            domain="separator",
            scale=Scalar(1)
        )
        sto_e_p = Variable(
            "Positive electrolyte stoichiometry",
            domain="positive electrode",
            scale=Scalar(1)
        )
        sto_e = pybamm.concatenation(sto_e_n, sto_e_sep, sto_e_p)
        phi_e_n = Variable(
            "Negative electrolyte potential [V]",
            domain="negative electrode",
            reference=-U_n_init
        )
        phi_e_sep = Variable(
            "Separator electrolyte potential [V]",
            domain="separator",
            reference=-U_n_init
        )
        phi_e_p = Variable(
            "Positive electrolyte potential [V]",
            domain="positive electrode",
            reference=-U_n_init
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_sep, phi_e_p)
        i_f_n = pybamm.Variable(
            "Negative lumped faradaic current [A]", domain="negative electrode"
        )
        i_f_p = pybamm.Variable(
            "Positive lumped faradaic current [A]", domain="positive electrode"
        )
        i_f = pybamm.concatenation(
            i_f_n, pybamm.PrimaryBroadcast(Scalar(0), "separator"), i_f_p
        )

        self.events += [
            Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_s_n_surf) - 0.01,
            ),
            Event(
                "Maximum negative particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_s_n_surf),
            ),
            Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(sto_s_p_surf) - 0.01,
            ),
            Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_s_p_surf),
            ),
        ]

        ######################
        # Parameters
        ######################
        F = self.param.F  # Faraday constant
        Rg = self.param.R  # Universal gas constant
        T = self.param.T_init  # Temperature
        RT_F = Rg * T / F  # Thermal voltage

        k_0_p = Parameter("Positive electrode lumped reaction rate constant [A]")
        k_0_n = Parameter("Negative electrode lumped reaction rate constant [A]")
        r_f_p = Parameter("Positive electrode lumped film resistance [Ohm]")
        r_f_n = Parameter("Negative electrode lumped film resistance [Ohm]")
        sigma_hat_p = Parameter("Positive electrode lumped solid conductivity [S]")
        sigma_hat_n = Parameter("Negative electrode lumped solid conductivity [S]")
        d_hat_p = FunctionParameter(
            "Positive electrode lumped solid diffusivity [s-1]",
            {"sto_s_p": sto_s_p, "T": T},
        )
        d_hat_n = FunctionParameter(
            "Negative electrode lumped solid diffusivity [s-1]",
            {"sto_s_n": sto_s_n, "T": T},
        )
        Q_param = Parameter("Cell total capacity [A.h]")
        kappa_hat_p = FunctionParameter(
            "Positive electrode lumped electrolyte conductivity [S]",
            {"sto_e_p": sto_e_p, "T": T},
        )
        kappa_hat_n = FunctionParameter(
            "Negative electrode lumped electrolyte conductivity [S]",
            {"sto_e_n": sto_e_n, "T": T},
        )
        kappa_hat_sep = FunctionParameter(
            "Separator lumped electrolyte conductivity [S]",
            {"sto_e_sep": sto_e_sep, "T": T},
        )
        kappa_hat = pybamm.concatenation(kappa_hat_n, kappa_hat_sep, kappa_hat_p)
        kappa_hat_D = FunctionParameter(
            "Electrolyte lumped constant for transport and thermodynamic factor [V.K-1]",
            {"sto_e": sto_e, "T": T},
        )
        psi_hat = FunctionParameter(
            "Scale ratio between lumped electrolyte diffusivity and conductivity [V.K-1]",
            {"sto_e": sto_e, "T": T},
        )
        q_e_p = FunctionParameter(
            "Positive electrode lumped quantity of electrolyte concentration [A.h]",
            {"sto_e_p": sto_e_p, "T": T},
        )
        q_e_n = FunctionParameter(
            "Negative electrode lumped quantity of electrolyte concentration [A.h]",
            {"sto_e_n": sto_e_n, "T": T},
        )
        q_e_sep = FunctionParameter(
            "Separator lumped quantity of electrolyte concentration [A.h]",
            {"sto_e_sep": sto_e_sep, "T": T},
        )
        q_e = pybamm.concatenation(q_e_n, q_e_sep, q_e_p)

        ######################
        # Input current (positive on discharge)
        ######################
        I = self.param.current_with_time

        ######################
        # State of Charge
        ######################
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        self.rhs[Qt] = abs(I) / 3600

        self.initial_conditions[Q] = Scalar(0)
        self.initial_conditions[Qt] = Scalar(0)

        ######################
        # Kinetics
        ######################
        alpha = 0.5  # cathodic transfer coefficient
        i_0_p = k_0_p * (
            sto_s_p_surf**alpha * (sto_e_p * (1 - sto_s_p_surf)) ** (1 - alpha)
        )
        i_0_n = k_0_n * (
            sto_s_n_surf**alpha * (sto_e_n * (1 - sto_s_n_surf)) ** (1 - alpha)
        )
        eta_p = phi_s_p - phi_e_p - U_p - r_f_p * i_f_p
        eta_n = phi_s_n - phi_e_n - U_n - r_f_n * i_f_n

        self.algebraic[i_f_p] = i_f_p - i_0_p * (
            pybamm.exp((1 - alpha) * eta_p / RT_F) - pybamm.exp(-alpha * eta_p / RT_F)
        )
        self.algebraic[i_f_n] = i_f_n - i_0_n * (
            pybamm.exp((1 - alpha) * eta_n / RT_F) - pybamm.exp(-alpha * eta_n / RT_F)
        )

        self.initial_conditions[i_f_n] = Scalar(0)
        self.initial_conditions[i_f_p] = Scalar(0)

        ######################
        # Charge conservation in solid
        ######################
        i_s_p = sigma_hat_p * pybamm.grad(phi_s_p)  # Sigma must be a constant! careful
        i_s_n = sigma_hat_n * pybamm.grad(phi_s_n)

        self.algebraic[phi_s_p] = pybamm.div(i_s_p) - i_f_p
        self.algebraic[phi_s_n] = pybamm.div(i_s_n) - i_f_n

        self.boundary_conditions[phi_s_p] = {
            "left": (Scalar(0), "Neumann"),
            "right": (-I / sigma_hat_p, "Neumann"),
        }
        self.boundary_conditions[phi_s_n] = {
            "left": (Scalar(0), "Dirichlet"),
            "right": (Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_s_p] = U_p_init - U_n_init
        self.initial_conditions[phi_s_n] = Scalar(0)

        ######################
        # Mass conservation in solid
        ######################
        self.rhs[sto_s_p] = pybamm.div(d_hat_p * pybamm.grad(sto_s_p))
        self.rhs[sto_s_n] = pybamm.div(d_hat_n * pybamm.grad(sto_s_n))

        d_hat_p_surf = pybamm.surf(d_hat_p)
        d_hat_n_surf = pybamm.surf(d_hat_n)

        self.boundary_conditions[sto_s_p] = {
            "left": (Scalar(0), "Neumann"),
            "right": (
                -i_f_p * abs(sto_s_p_100 - sto_s_p_0) / 10800 / Q_param / d_hat_p_surf,
                "Neumann",
            ),
        }
        self.boundary_conditions[sto_s_n] = {
            "left": (Scalar(0), "Neumann"),
            "right": (
                -i_f_n * abs(sto_s_n_100 - sto_s_n_0) / 10800 / Q_param / d_hat_n_surf,
                "Neumann",
            ),
        }

        self.initial_conditions[sto_s_p] = pybamm.FullBroadcast(
            sto_s_p_init,
            "positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        self.initial_conditions[sto_s_n] = pybamm.FullBroadcast(
            sto_s_n_init,
            "negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )

        ######################
        # Charge conservation in electrolyte
        ######################
        sto_e_safe = pybamm.maximum(sto_e, Scalar(1e-6))
        i_e = kappa_hat * (
            pybamm.grad(phi_e) + kappa_hat_D * T * pybamm.grad(sto_e) / sto_e_safe
        )

        i_D_left = (
            pybamm.boundary_value(kappa_hat_D, "left")
            * T
            * pybamm.boundary_gradient(sto_e, "left")
            / pybamm.boundary_value(sto_e_safe, "left")
        )
        i_D_right = (
            pybamm.boundary_value(kappa_hat_D, "right")
            * T
            * pybamm.boundary_gradient(sto_e, "right")
            / pybamm.boundary_value(sto_e_safe, "right")
        )

        self.algebraic[phi_e] = pybamm.div(i_e) + i_f

        self.boundary_conditions[phi_e] = {
            "left": (-i_D_left, "Neumann"),
            "right": (-i_D_right, "Neumann"),
        }
        self.initial_conditions[phi_e] = -U_n_init

        ######################
        # Mass conservation in electrolyte
        ######################
        self.rhs[sto_e] = (
            (psi_hat * T * pybamm.div(kappa_hat * pybamm.grad(sto_e)) + i_f)
            / 3600
            / q_e
        )
        self.boundary_conditions[sto_e] = {
            "left": (Scalar(0), "Neumann"),
            "right": (Scalar(0), "Neumann"),
        }

        self.initial_conditions[sto_e] = Scalar(1)

        ######################
        # Cell voltage
        ######################
        V = pybamm.boundary_value(phi_s_p, "right") - pybamm.boundary_value(
            phi_s_n, "left"
        )
        # Save the initial OCV
        self.param.ocv_init = U_p_init - U_n_init

        # Events specify points at which a solution should terminate
        self.events += [
            Event("Minimum voltage [V]", V - self.param.voltage_low_cut),
            Event("Maximum voltage [V]", self.param.voltage_high_cut - V),
        ]

        ######################
        # (Some) variables
        ######################
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative particle stoichiometry": sto_s_n,
            "Negative particle surface stoichiometry": sto_s_n_surf,
            "Positive particle stoichiometry": sto_s_p,
            "Positive particle surface stoichiometry": sto_s_p_surf,
            "Negative electrolyte stoichiometry": sto_e_n,
            "Separator electrolyte stoichiometry": sto_e_sep,
            "Positive electrolyte stoichiometry": sto_e_p,
            "Electrolyte stoichiometry": sto_e,
            "Negative electrode potential [V]": phi_s_n,
            "Positive electrode potential [V]": phi_s_p,
            "Negative electrolyte potential [V]": phi_e_n,
            "Positive electrolyte potential [V]": phi_e_p,
            "Separator electrolyte potential [V]": phi_e_sep,
            "Electrolyte potential [V]": phi_e,
            "Negative lumped faradaic current [A]": i_f_n,
            "Positive lumped faradaic current [A]": i_f_p,
            "Lumped faradaic current [A]": i_f,
            "Time [s]": pybamm_t,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Discharge capacity [A.h]": Q,
            "Throughput capacity [A.h]": Qt,
            "Voltage [V]": V,
            "Battery voltage [V]": V,
            "Open-circuit voltage [V]": pybamm.x_average(U_p) - pybamm.x_average(U_n),
        }

    def U(self, sto, domain):
        """
        Dimensional open-circuit potential [V], calculated as U(x) = U_ref(x).
        Credit: PyBaMM
        """
        # bound stoichiometry between tol and 1-tol. Adding 1/sto + 1/(sto-1) later
        # will ensure that ocp goes to +- infinity if sto goes into that region
        # anyway
        Domain = domain.capitalize()
        tol = pybamm.settings.tolerances["U__c_s"]
        sto = pybamm.maximum(pybamm.minimum(sto, 1 - tol), tol)
        inputs = {f"{Domain} particle surface stoichiometry": sto}
        u_ref = FunctionParameter(f"{Domain} electrode OCP [V]", inputs)

        # add a term to ensure that the OCP goes to infinity at 0 and -infinity at 1
        # this will not affect the OCP for most values of sto
        out = u_ref + 1e-6 * (1 / sto + 1 / (sto - 1))

        if domain == "negative":
            out.print_name = r"U_\mathrm{n}(c^\mathrm{surf}_\mathrm{s,n})"
        elif domain == "positive":
            out.print_name = r"U_\mathrm{p}(c^\mathrm{surf}_\mathrm{s,p})"
        return out

    def build_model(self):
        """
        Build model variables and equations
        Credit: PyBaMM
        """
        self._build_model()

        self._built = True
        pybamm.logger.info(f"Finish building {self.name}")

    @property
    def default_parameter_values(self) -> ParameterValues:
        param = ParameterValues("Chen2020")
        # ce0 = param["Initial concentration in electrolyte [mol.m-3]"]
        # T = param["Ambient temperature [K]"]
        # param["Electrolyte conductivity [S.m-1]"] = param[
        #     "Electrolyte conductivity [S.m-1]"
        # ](ce0, T)
        # param["Electrolyte diffusivity [m2.s-1]"] = param[
        #     "Electrolyte diffusivity [m2.s-1]"
        # ](ce0, T)
        return self.create_grouped_parameters(param)

    @property
    def default_quick_plot_variables(self):
        return [
            "Negative particle surface stoichiometry",
            "Electrolyte stoichiometry",
            "Positive particle surface stoichiometry",
            "Current [A]",
            {
                "Negative electrode potential [V]",
            },
            "Electrolyte potential [V]",
            {
                "Positive electrode potential [V]",
            },
            {"Open-circuit voltage [V]", "Voltage [V]"},
        ]

    @property
    def default_var_pts(self):
        x_n = SpatialVariable(
            "x_n",
            domain=["negative electrode"],
            coord_sys="cartesian",
        )
        x_s = SpatialVariable(
            "x_s",
            domain=["separator"],
            coord_sys="cartesian",
        )
        x_p = SpatialVariable(
            "x_p",
            domain=["positive electrode"],
            coord_sys="cartesian",
        )

        # Add particle domains
        r_n = SpatialVariable(
            "r_n",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
            coord_sys="spherical polar",
        )
        r_p = SpatialVariable(
            "r_p",
            domain=["positive particle"],
            auxiliary_domains={"secondary": "positive electrode"},
            coord_sys="spherical polar",
        )

        return {x_n: 20, x_s: 20, x_p: 20, r_n: 20, r_p: 20}

    @property
    def default_geometry(self):
        return {
            "negative electrode": {"x_n": {"min": 0, "max": 1}},
            "separator": {"x_s": {"min": 1, "max": 2}},
            "positive electrode": {"x_p": {"min": 2, "max": 3}},
            "negative particle": {"r_n": {"min": 0, "max": 1}},
            "positive particle": {"r_p": {"min": 0, "max": 1}},
        }

    @property
    def default_submesh_types(self):
        return {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }

    @property
    def default_spatial_methods(self):
        return {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
            "positive electrode": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }

    @staticmethod
    def create_grouped_parameters(parameter_values: ParameterValues) -> ParameterValues:
        """
        Create a parameter set for the Grouped Single Particle Model with Electrolyte from a
        PyBaMM lithium-ion ParameterValues object.

        Parameters
        ----------
        parameter_values : pybamm.ParameterValues
            Parameters and their corresponding values.

        Returns
        -------
        parameter_values : pybamm.ParameterValues
            A new set of parameters and their values.
        """
        param = parameter_values

        # Unpack physical parameters
        F = pybamm.constants.F.value
        R = pybamm.constants.R.value
        T = param["Ambient temperature [K]"]
        I = param["Current function [A]"]
        eps_s_p = param["Positive electrode active material volume fraction"]
        eps_s_n = param["Negative electrode active material volume fraction"]
        c_max_p = param["Maximum concentration in positive electrode [mol.m-3]"]
        c_max_n = param["Maximum concentration in negative electrode [mol.m-3]"]
        L_p = param["Positive electrode thickness [m]"]
        L_n = param["Negative electrode thickness [m]"]
        L_sep = param["Separator thickness [m]"]
        eps_e_p = param["Positive electrode porosity"]
        eps_e_n = param["Negative electrode porosity"]
        eps_e_sep = param["Separator porosity"]
        R_p = param["Positive particle radius [m]"]
        R_n = param["Negative particle radius [m]"]
        D_p = param["Positive particle diffusivity [m2.s-1]"]
        D_n = param["Negative particle diffusivity [m2.s-1]"]
        b_e_p = param["Positive electrode Bruggeman coefficient (electrolyte)"]
        b_e_n = param["Negative electrode Bruggeman coefficient (electrolyte)"]
        b_e_sep = param["Separator Bruggeman coefficient (electrolyte)"]
        b_s_p = param["Positive electrode Bruggeman coefficient (electrode)"]
        b_s_n = param["Negative electrode Bruggeman coefficient (electrode)"]
        i_0_p = 3.42e-6  # (A/m2)(m3/mol)**1.5
        i_0_n = 6.48e-7  # (A/m2)(m3/mol)**1.5
        rSEI = param["SEI resistivity [Ohm.m]"]
        dSEI = param["Initial SEI thickness [m]"]
        sigma_p = param["Positive electrode conductivity [S.m-1]"] * eps_s_p**b_s_p
        sigma_n = param["Negative electrode conductivity [S.m-1]"] * eps_s_n**b_s_n

        ce0 = param["Initial concentration in electrolyte [mol.m-3]"]
        t_plus = param["Cation transference number"]

        kappa_e_fun = param["Electrolyte conductivity [S.m-1]"] 
        D_e_fun = param["Electrolyte diffusivity [m2.s-1]"]
        A = param["Electrode height [m]"] * param["Electrode width [m]"]

        def kappa_hat_n(sto_e_n, T):
            ce = ce0 * sto_e_n
            return kappa_e_fun(ce, T) * eps_e_n**b_e_n * A / L_p
        def kappa_hat_p(sto_e_p, T):
            ce = ce0 * sto_e_p
            return kappa_e_fun(ce, T) * eps_e_p**b_e_p * A / L_p
        def kappa_hat_sep(sto_e_sep, T):
            ce = ce0 * sto_e_sep
            return kappa_e_fun(ce, T) * eps_e_sep**b_e_sep * A / L_sep

        def psi_hat(sto_e, T):
            ce = ce0 * sto_e
            return F * D_e_fun(ce, T) * ce0 / ((1 - t_plus) * kappa_e_fun(ce, T) * T)
        a_s_p = 3.0 * eps_s_p / R_p
        a_s_n = 3.0 * eps_s_n / R_n
        k_0_p_phys = i_0_p / F
        k_0_n_phys = i_0_n / F
        k_0_p_norm = k_0_p_phys * c_max_p * ce0**0.5
        k_0_n_norm = k_0_n_phys * c_max_n * ce0**0.5
        r_f_p_phys = 0  # [Ohm.m2]
        r_f_n_phys = rSEI * dSEI

        # Compute grouped parameters
        sto_s_n_0, sto_s_n_100, sto_s_p_100, sto_s_p_0 = get_min_max_stoichiometries(
            param
        )
        sto_p_init = (
            param["Initial concentration in positive electrode [mol.m-3]"] / c_max_p
        )
        soc_init = (sto_p_init - sto_s_p_0) / (sto_s_p_100 - sto_s_p_0)
        k_0_p = a_s_p * A * L_p * F * k_0_p_norm
        k_0_n = a_s_n * A * L_n * F * k_0_n_norm
        r_f_p = r_f_p_phys / a_s_p / A / L_p
        r_f_n = r_f_n_phys / a_s_n / A / L_n
        sigma_hat_p = sigma_p * A / L_p
        sigma_hat_n = sigma_n * A / L_n
        d_hat_p = D_p / R_p**2
        d_hat_n = D_n / R_n**2
        Q_param = param["Nominal cell capacity [A.h]"]
        kappa_hat_D = 2 * R * (t_plus - 1) / F
        q_e_p = eps_e_p * ce0 * A * L_p * F / 3600 / (1 - t_plus)
        q_e_n = eps_e_n * ce0 * A * L_n * F / 3600 / (1 - t_plus)
        q_e_sep = eps_e_sep * ce0 * A * L_sep * F / 3600 / (1 - t_plus)

        parameter_dictionary = {
            "Nominal cell capacity [A.h]": Q_param,
            "Cell total capacity [A.h]": Q_param,
            "Current function [A]": I,
            "Initial temperature [K]": T,
            "Initial SoC": soc_init,
            "Minimum negative stoichiometry": sto_s_n_0,
            "Maximum negative stoichiometry": sto_s_n_100,
            "Minimum positive stoichiometry": sto_s_p_100,
            "Maximum positive stoichiometry": sto_s_p_0,
            "Lower voltage cut-off [V]": param["Lower voltage cut-off [V]"],
            "Upper voltage cut-off [V]": param["Upper voltage cut-off [V]"],
            "Positive electrode OCP [V]": param["Positive electrode OCP [V]"],
            "Negative electrode OCP [V]": param["Negative electrode OCP [V]"],
            "Positive electrode lumped reaction rate constant [A]": k_0_p,
            "Negative electrode lumped reaction rate constant [A]": k_0_n,
            "Positive electrode lumped film resistance [Ohm]": r_f_p,
            "Negative electrode lumped film resistance [Ohm]": r_f_n,
            "Positive electrode lumped solid conductivity [S]": sigma_hat_p,
            "Negative electrode lumped solid conductivity [S]": sigma_hat_n,
            "Positive electrode lumped solid diffusivity [s-1]": d_hat_p,
            "Negative electrode lumped solid diffusivity [s-1]": d_hat_n,
            "Positive electrode lumped electrolyte conductivity [S]": kappa_hat_p,
            "Negative electrode lumped electrolyte conductivity [S]": kappa_hat_n,
            "Separator lumped electrolyte conductivity [S]": kappa_hat_sep,
            "Electrolyte lumped constant for transport and thermodynamic factor [V.K-1]": kappa_hat_D,
            "Scale ratio between lumped electrolyte diffusivity and conductivity [V.K-1]": psi_hat,
            "Positive electrode lumped quantity of electrolyte concentration [A.h]": q_e_p,
            "Negative electrode lumped quantity of electrolyte concentration [A.h]": q_e_n,
            "Separator lumped quantity of electrolyte concentration [A.h]": q_e_sep,
        }
        parameter_values = ParameterValues(values=parameter_dictionary)
        parameter_values._set_initial_state = set_initial_state  # noqa: SLF001
        return parameter_values


def set_initial_state(
    initial_value,
    parameter_values,
    direction=None,
    param=None,
    inplace=True,
    options=None,
    inputs=None,
    tol=1e-6,
):
    """
    Set the value of the initial state of charge.

    Parameters
    ----------
    initial_value : float
        Target initial value.
        If float, interpreted as SOC, must be between 0 and 1.
        If string e.g. "4 V", interpreted as voltage, must be between V_min and V_max.
    parameter_values : :class:`pybamm.ParameterValues`
        Parameters and their corresponding values.
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.
    inplace: bool, optional
        If True, replace the parameters values in place. Otherwise, return a new set of
        parameter values. Default is True.
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.
    inputs : dict, optional
        A dictionary of input parameters to pass to the model when solving.
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        A lower value results in higher precision but may increase computation time.
        Default is 1e-6.
    """
    parameter_values = parameter_values if inplace else parameter_values.copy()

    if isinstance(initial_value, int | float):
        if not 0 <= initial_value <= 1:
            raise ValueError("Initial SOC should be between 0 and 1")
        parameter_values["Initial SoC"] = initial_value

    else:
        raise ValueError("Initial value must be a float between 0 and 1.")

    return parameter_values
