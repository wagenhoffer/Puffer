## Panels and Unsteady Flow - Numerical Methodologies 

A potential flow boundary element method is coupled to a transient acoustics boundary element method via Powell's acoustic analogy to compute all hydrodynamic and acoustic quantities in this work. The methodology and validation of the framework are presented in full detail by Wagenhoffer *et al.* [1,2].

The potential flow solver is an adaptation of the panel method described by Willis *et al.* [3]. The inviscid flow around hydrofoils can be found by solving the Laplace equations with an imposed no-penetration boundary condition on the body surface,

```math
\nabla \phi \cdot \mathbf{\hat{n}} = 0 ~~ {\rm on}~~ S_{\rm{b}},
```

where $\mathbf{\hat{n}}$ is the outward normal of the surface. The boundary integral equation integrates the effects of the combined distribution of sources and doublets on the body surface $S_{\rm{b}}$ and doublets on edge panel $S_{\rm{e}}$, with vortex particles in the wake. The scalar potential may be written as

```math
\phi(\mathbf{x}) = \int_{S_{\rm{b}}} \left[\sigma(\mathbf{y}) G(\mathbf{x},\mathbf{y}) - \mu(\mathbf{y})\hat{\mathbf{n}}\cdot \nabla G(\mathbf{x},\mathbf{y})\right]dS -\int_{S_{\rm{e}}}\mu_{\rm e}(\mathbf{y})\hat{\mathbf{n}}\cdot \nabla G(\mathbf{x},\mathbf{y})dS,
```

where $\mathbf{y}$ is a source position, $\mathbf{x}$ is the observer location, and $G(\mathbf{x},\mathbf{y})= \frac{1}{2\pi}\ln|\mathbf{x}-\mathbf{y}|$ is the two-dimensional Green's function for the Laplace equation. The source and doublet strengths are defined respectively as

```math
\begin{align*}
\sigma &= \hat{\mathbf{n}} \cdot ( \mathbf{U} + \mathbf{U}_{\rm{rel}} - \mathbf{U}_\omega),\\
\mu &= \phi_i - \phi,
\end{align*}
```

where $U_\omega$ is velocity induced by the vortex particles in the field, $\mathbf{U}$ is the body velocity, $\mathbf{U}_{\rm rel}$ is the velocity of the center of each element relative to the body-frame of reference, and $\phi_i = 0$ is the interior potential of the body.

At each time step, vorticity is defined at the trailing edge to satisfy the Kutta condition. The trailing edge panel is assigned the potential difference between the upper and lower panels at the trailing edge of the foil, $\mu_{\rm e} = \mu_{\rm{upper}}-\mu_{\rm{lower}}$, ensuring that the trailing edge of the hydrofoil has no bound circulation. Vorticity is represented in the computational domain by discrete, radially-symmetric, desingularized Gaussian vortex particles. The induced velocity of the vortex blobs is evaluated by application of the Biot-Savart law, yielding [4]:

```math
\mathbf{U}_\omega(\mathbf{x},t) = \sum_{i=1}^N \frac{\Gamma_i}{2\pi} \left[1-\exp\left(\frac{-|\mathbf{x - y}|}{2{r_{\rm{cut}}}^2}\right)\right],
```

where $\Gamma$ is the circulation of the vortex particle and $r_{\rm{cut}}$ is the cut-off radius. Following the work of Pan *et al.* [5], the cut-off radius is set to $r_{\rm{cut}} = 1.3\Delta t$ for time step $\Delta t$ to ensure that the wake particle cores overlap and a thin vortex sheet is shed. The evolution of the vortex particle position is updated using a forward Euler scheme [3]. The use of discrete vortices to represent the wake requires the use of two edge panels set behind the foil. The first edge panel, set with the empirical length of $l_{\rm{panel}} = 0.4 U_{\infty}\Delta t$ [6], satisfies the Kutta condition at the trailing edge. Next, the buffer panel is attached to the edge panel and stores information about the previous time step.

The induced velocity of the vortex particles on the body is accounted for in the definition of the source strength $\sigma$. The vortex particle induced velocity also augments the pressure calculation put forth by Katz and Plotkin [6]. The surface pressure is determined by

```math
\frac{P_\infty - P(x)}{\rho} = \left.\frac{\partial \phi_{\rm{wake}}}{\partial t}\right|_{\rm{body}} +\left.\frac{\partial \phi}{\partial t}\right|_{\rm{body}} - (\mathbf{U}+\mathbf{U}_{\rm rel})\cdot(\nabla \phi + \mathbf{U}_\omega) +\frac{1}{2}|\nabla \phi + \mathbf{U}_\omega|^2,
```

where $\partial \phi_{\rm{wake}}/\partial t = \Gamma \dot{\theta}/(2\pi)$ is the time rate of change due to a vortex particle with circulation $\Gamma$ at an angle $\theta$ from the observation point, and $\rho$ is the fluid density. The pressure found on the body in (\ref{eqn:flow-pressure}) is similar to the form put forth by Willis *et al.* [3], but here $\partial \phi_{\rm{wake}}/\partial t$ is the positional change of a vortex with respect to a panel and does not require the solution of a secondary system to find the influence of wake vortices onto the body surface.

[1] Wagenhoffer, Nathan, Keith W. Moored, and Justin W. Jaworski. "Unsteady propulsion and the acoustic signature of undulatory swimmers in and out of ground effect." Physical Review Fluids 6.3 (2021): 033101. https://doi.org/10.1103/PhysRevFluids.6.033101

[2] Wagenhoffer, Nathan, Keith W. Moored, and Justin W. Jaworski. "On the noise generation and unsteady performance of combined heaving and pitching foils." Bioinspiration & Biomimetics 18.4 (2023): 046011. https://doi.org/10.1088/1748-3190/acd59d

[3] Willis, D. J., Peraire, J., and White, J. K., “A combined pFFT–multipole tree code, unsteady panel method with vortex particle wakes,” International Journal for Numerical Methods in Fluids, Vol. 53, No. 8, 2007, pp. 1399–1422. 

[4] Cottet, G.-H. and Koumoutsakos, P. D., Vortex methods: theory and practice, Cambridge university press, 2000.

[5] Y. Pan, X. Dong, Q. Zhu, and D. K. Yue, “Boundary-element method for the prediction of performance of flapping foils with leading-edge separation,” Journal of FluidMechanics 698, 446–467 (2012).

[6] Katz, J. and Plotkin, A., Low-speed aerodynamics, Vol. 13, Cambridge university press, 2001.
