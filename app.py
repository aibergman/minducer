"""Hugging Face / Gradio entry point for the Induced-Moment Exchange Explorer."""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ZeroGPU Spaces provides this package at runtime. Keep local CPU-only
# development import-safe when the package is not installed.
try:
    import spaces
except ImportError:  # pragma: no cover - depends on the hosting runtime
    class _LocalSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorate(function):
                return function

            return decorate

    spaces = _LocalSpaces()


# Hugging Face ZeroGPU requires at least one GPU-decorated function at startup.
# The explorer itself is CPU-only; this probe is deliberately not connected to
# any UI event and is never called during normal use.
@spaces.GPU(duration=1)
def _zerogpu_probe():
    return True


# The Space runs ``python app.py`` directly from a source-layout repository.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Gradio imports a plotting component during startup.  Keep Matplotlib's
# cache in a writable ephemeral location on minimal Space containers.
MPL_CACHE = Path(tempfile.gettempdir()) / "induced_exchange_mpl"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

APP_CSS = """
:root { --imx-ink: #183b56; --imx-teal: #18a999; --imx-coral: #e07a5f; --imx-paper: #f7fafc; --imx-page: #102a43; --imx-panel: #183b56; --imx-text-muted: #c7d6df; color-scheme: dark; }
html, body, .gradio-container { background: var(--imx-page) !important; color: var(--imx-paper) !important; }
.gradio-container {
  --body-background-fill: var(--imx-page) !important;
  --background-fill-primary: var(--imx-panel) !important;
  --background-fill-secondary: var(--imx-page) !important;
  --body-text-color: var(--imx-paper) !important;
  --body-text-color-subdued: var(--imx-text-muted) !important;
  --block-background-fill: var(--imx-panel) !important;
  --block-info-text-color: var(--imx-text-muted) !important;
  --block-label-background-fill: var(--imx-teal) !important;
  --block-label-text-color: var(--imx-page) !important;
  --block-title-text-color: var(--imx-page) !important;
  --checkbox-label-text-color: var(--imx-paper) !important;
  --table-text-color: var(--imx-paper) !important;
  --table-even-background-fill: #214b63 !important;
  --table-odd-background-fill: var(--imx-panel) !important;
  --button-secondary-background-fill: #214b63 !important;
  --button-secondary-text-color: var(--imx-paper) !important;
  --input-background-fill: var(--imx-paper) !important;
  --input-background-fill-hover: #ffffff !important;
  --input-background-fill-focus: #ffffff !important;
  --input-text-color: #000000 !important;
  --input-placeholder-color: #597080 !important;
}
.gradio-container .gradio-dropdown input,
.gradio-container .gradio-dropdown .wrap,
.gradio-container .gradio-dropdown .options,
.gradio-container .gradio-dropdown .options *,
.gradio-container .gradio-dropdown [role="listbox"],
.gradio-container .gradio-dropdown [role="listbox"] * {
  color: #000000 !important;
}
/* DropdownOptions is rendered as a global fixed-position menu in Gradio. */
ul.options,
ul.options *,
[role="option"],
[role="option"] * {
  color: #000000 !important;
}
input[role="listbox"] {
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
}
ul.options,
ul.options .item,
ul.options [role="option"] {
  background-color: #ffffff !important;
}
ul.options .item:hover,
ul.options .item.active,
ul.options [role="option"]:hover,
ul.options [role="option"][aria-selected="true"] {
  background-color: #f0f0f0 !important;
}
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose blockquote,
.gradio-container .label-wrap label,
.gradio-container .block-info,
.gradio-container .info,
.gradio-container .description,
.gradio-container .tab-nav button { color: var(--imx-paper) !important; }
.gradio-container .imx-plot-heading { color: var(--imx-teal) !important; }
.gradio-container .imx-file-upload .wrap { color: var(--imx-paper) !important; }
.gradio-container .imx-file-upload .wrap .or { color: var(--imx-text-muted) !important; }
.gradio-container .imx-file-upload .wrap .icon-wrap { color: var(--imx-text-muted) !important; }
.imx-shell { max-width: 1380px; margin: 0 auto; }
.imx-hero { background: linear-gradient(125deg, #183b56 0%, #236477 58%, #18a999 100%); color: white; border-radius: 20px; padding: 30px 34px; margin-bottom: 18px; box-shadow: 0 14px 32px rgba(24,59,86,.16); }
.imx-hero h1 { margin: 0 0 8px; font-size: 2.2rem; letter-spacing: -.03em; }
.imx-hero p { margin: 0; max-width: 860px; color: #e7f4f2; font-size: 1.05rem; }
.imx-kicker { text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 700; color: #a9ebe0; }
.imx-note { border-left: 4px solid var(--imx-coral); background: #fff7f3; padding: 12px 15px; border-radius: 8px; color: #5b3b35; }
.imx-card { border: 1px solid #dfe8ed; background: rgba(255,255,255,.72); border-radius: 14px; padding: 6px; }
.imx-muted { color: #597080; }
.imx-plot-heading { color: var(--imx-teal); font-size: .95rem; font-weight: 700; margin: 4px 0 6px; padding: 0 6px; }
.imx-plot-heading p { margin: 0; }
footer { display: none !important; }
"""


def _configure_launch_options(demo, gr):
    """Apply theme/CSS across Gradio versions around the 5-to-6 transition."""

    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")
    parameters = inspect.signature(demo.launch).parameters
    options = {}
    if "theme" in parameters:
        options["theme"] = theme
    else:
        # Gradio 5.50 still consumes these values from the Blocks instance,
        # but warns when they are supplied to Blocks(...).  Configure the
        # already-built instance to avoid both the warning and a launch
        # TypeError on versions that have not moved the arguments yet.
        demo.theme = theme
        demo.theme_css = theme._get_theme_css()
        demo.stylesheets = theme._stylesheets
        demo.theme_hash = hashlib.sha256(demo.theme_css.encode("utf-8")).hexdigest()
    if "css" in parameters:
        options["css"] = APP_CSS
    else:
        demo.css = APP_CSS
    return options


def build_demo():
    """Build the UI lazily so the scientific package remains Gradio-free."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised on bare library installs
        raise RuntimeError("The Space UI needs Gradio. Install the dependencies from requirements.txt.") from exc

    from induced_exchange.space import (
        UploadMappingError,
        AnalysisSession,
        _figure_dressed,
        _figure_dressed_realspace,
        _figure_dressed_realspace_delta,
        _figure_dressed_realspace_relative,
        _figure_exchange,
        _figure_magnons,
        _figure_realspace,
        _figure_response,
        analyse_model,
        classification_from_induced_sites,
        classification_rows,
        cleanup_analysis_artifacts,
        default_induced_sites,
        default_stiffness_q_max,
        dressed_shell_rows,
        export_files,
        inference_rows,
        inspect_upload_mapping,
        load_uploaded_set,
        magnon_markdown,
        model_summary,
        ordering_markdown,
        response_markdown,
        shell_rows,
    )
    from induced_exchange.io_uppasd import InputFormatError, LoadedUppASD

    examples_root = Path(__file__).resolve().parent / "examples"
    example_choices = ["Upload my files", "Simple analytic FM", "Induced-moment toy", "FePt-style UppASD", "UppASD symmetry-reduced Jij"]
    example_paths = {
        "Simple analytic FM": examples_root / "simple_fm" / "inpsd.dat",
        "Induced-moment toy": examples_root / "induced_toy" / "inpsd.dat",
        "FePt-style UppASD": examples_root / "fept_style" / "inpsd.dat",
        "UppASD symmetry-reduced Jij": examples_root / "uppasd_style" / "inpsd.dat",
    }

    def plot_heading(text):
        """Render plot context outside the Matplotlib canvas."""

        return gr.Markdown(f"<div class='imx-plot-heading'>{text}</div>")

    with gr.Blocks(title="Induced-Moment Exchange Explorer") as demo:
        session_state = gr.State(None)
        gr.HTML("<div class='imx-hero'><div class='imx-kicker'>Scientific exchange analysis · CPU-friendly</div><h1>Induced-Moment Exchange Explorer</h1><p>From UppASD-style Jij input to reciprocal exchange, induced response, downfolded interactions, and FM-compatible magnon diagnostics.</p></div>", padding=False)

        with gr.Tabs():
            with gr.Tab("1 · Input"):
                gr.Markdown("### Bring an UppASD input set into the explorer\nUpload `inpsd.dat` and its referenced files. The `inpsd.dat` file is required because it supplies the explicit cell vectors used for reciprocal-space analysis; individual `posfile`, `momfile`, and `jfile` uploads alone are not loaded and no identity cell is assumed. If `inpsd.dat` contains `alat` (in metres), the loader applies it automatically to the cell, positions, and exchange displacements. References are matched by basename because browser uploads do not preserve arbitrary relative paths; manual assignment is available below.")
                with gr.Row():
                    example = gr.Dropdown(example_choices, value="Upload my files", label="Starting point")
                    energy_unit = gr.Dropdown(["mRy", "meV", "eV", "unspecified"], value="mRy", label="Energy unit (default: mRy; confirm input)")
                    symmetry_mode = gr.Radio(
                        [("Complete jfile (use as supplied)", "complete"), ("Symmetry-reduced jfile (expand with spglib)", "reduced")],
                        value="complete",
                        label="Jij symmetry",
                        info="Complete accepts a full UppASD neighbour setup. Reduced generates symmetry-related neighbours; conflicting Jij values are reported, never averaged.",
                    )
                with gr.Row():
                    inpsd_file = gr.File(label="inpsd.dat", file_count="single", type="filepath", elem_classes=["imx-file-upload"])
                    supporting_files = gr.File(label="posfile / momfile / jfile (max 32 files / 100 MiB)", file_count="multiple", type="filepath", elem_classes=["imx-file-upload"])
                with gr.Row():
                    refresh_mapping = gr.Button("Inspect file mapping", variant="secondary")
                    load_button = gr.Button("Load input set", variant="primary")
                mapping_table = gr.Dataframe(headers=["reference", "uploaded basename", "status"], datatype=["str", "str", "str"], row_count=1, label="Referenced-file mapping", interactive=False)
                with gr.Row():
                    manual_pos = gr.Dropdown([], label="Manual posfile assignment", allow_custom_value=False)
                    manual_mom = gr.Dropdown([], label="Manual momfile assignment", allow_custom_value=False)
                    manual_exchange = gr.Dropdown([], label="Manual exchange assignment", allow_custom_value=False)
                load_status = gr.Markdown()
                parsed_summary = gr.Markdown()
                classification = gr.Dataframe(headers=["site", "atom_type", "moment (mu_B)", "role"], datatype=["number", "str", "number", "str"], type="array", row_count=1, label="Role preview (default: m < 0.5 μB is induced)", interactive=False)
                induced_site_toggle = gr.CheckboxGroup(choices=[], value=[], label="Induced sites (toggle)", info="Select induced sites explicitly. By default, sites with reference moment m < 0.5 μB are selected; all other sites are robust.")
                with gr.Row():
                    mesh_size = gr.Slider(4, 16, value=8, step=1, label="Seekpath points per segment", info="The exchange, response, dressed-exchange, and magnon plots use this high-symmetry path. The same value controls the auxiliary 3-D mesh used only for J_eff(r).")
                    analyze_button = gr.Button("Run analysis", variant="primary")

            with gr.Tab("2 · Raw exchange"):
                gr.Markdown("### Raw rigid-site exchange\nAll supplied sites and literal UppASD jfile Jij rows are retained. The ordering diagnostic is the largest eigenvalue of J(q) under the UppASD ordered-pair convention `H = −Σᵢ≠ⱼ Jij eᵢ·eⱼ`. The scan is restricted to the seekpath high-symmetry path (or the explicitly labelled fallback path if seekpath cannot classify the structure); it is not a global 3-D ordering search. The distance plot separates `J_MM` (robust–robust, circles), `J_Mm` (robust–induced, squares), and `J_mm` (induced–induced, triangles).")
                raw_ordering = gr.Markdown("Load an input set and run analysis.")
                with gr.Row():
                    with gr.Column():
                        plot_heading("Raw Jij by subspace · distance")
                        distance_plot = gr.Plot(label="Raw Jij by subspace · distance", show_label=False)
                    with gr.Column():
                        plot_heading("J(q) eigenvalue scan · high-symmetry path")
                        raw_plot = gr.Plot(label="J(q) eigenvalue scan · high-symmetry path", show_label=False)
                shell_table = gr.Dataframe(headers=["shell", "radius", "count", "mean Jij", "min Jij", "max Jij"], row_count=1, label="Radial exchange shells", interactive=False)

            with gr.Tab("3 · Induced response"):
                gr.Markdown("### Induced response\nClassify sites explicitly in the Input tab. The induced response is algebraic and instantaneous; induced sites are not independent LLG/LSWT degrees of freedom. The response panel uses a coherent unit-amplitude robust spin spiral along the same seekpath high-symmetry path.")
                with gr.Row():
                    response_mode = gr.Radio([("J-weighted approximation", "j_weighted"), ("Historical unweighted", "historical")], value="j_weighted", label="Response model")
                    x_override = gr.Textbox(label="X override", placeholder="blank = infer · e.g. 0.12 or 2:0.12, 3:0.08", info="J-weighted mode: X has inverse-energy units and p_ind = X K e, where p_ind = m_ind/|m_ind⁰|.")
                    include_induced = gr.Checkbox(value=True, label="Include induced–induced propagation")
                response_recompute = gr.Button("Recompute response and dressing", variant="primary")
                response_info = gr.Markdown()
                with gr.Row():
                    with gr.Column():
                        plot_heading("p_ind(q) / p_ind(Γ) · coherent spiral")
                        response_plot = gr.Plot(label="p_ind(q) / p_ind(Γ) · coherent spiral", show_label=False)
                    inference_table = gr.Dataframe(headers=["site", "m_ref (mu_B)", "source field (energy)", "X (1/energy)", "warnings"], row_count=1, label="X inference and source-field normalization", interactive=False)

            with gr.Tab("4 · Dressed exchange"):
                gr.Markdown("### Robust-space exchange after induced-moment elimination\nThe induced contribution is a model-dependent correction using the selected response approximation. `J_eff(q)` is the Fourier-space matrix in the robust basis; the q-plot shows its leading eigenvalue. For one robust site (such as Fe with Pt treated as induced), this matrix is a scalar. The first real-space plot uses shell means to compare the direct `J_RR(r)`, cross-block `K_RI(r)`, Mryasov/downfolded `J_Mryasov(r)`, and Polesya-like induced `J_Polesya(r)` channels. With the same static response model, the last two are numerically equivalent and are shown with separate labels. The induced correction `ΔJ_induced(r)` and its relative value `ΔJ_induced(r) / J_RR(r)` are plotted separately below. Real-space x-axes use the implicit UppASD lattice parameter convention (`alat = 1`), while all curves are reconstructed from a separate complete q mesh; moment normalization remains separate in the magnon calculation. The exported `dressed_jfile` is directly compatible with UppASD's ordered-pair convention.")
                dressed_info = gr.Markdown()
                with gr.Row():
                    with gr.Column():
                        plot_heading("Robust-space J_eff(q) · high-symmetry path")
                        dressed_plot = gr.Plot(label="Robust-space J_eff(q) · high-symmetry path", show_label=False)
                    with gr.Column():
                        plot_heading("Real-space exchange channels · shell means")
                        dressed_realspace_plot = gr.Plot(label="Real-space exchange channels · shell means", show_label=False)
                with gr.Row():
                    with gr.Column():
                        plot_heading("Induced correction ΔJ_induced(r) · shell means")
                        dressed_delta_realspace_plot = gr.Plot(label="Induced correction ΔJ_induced(r) · shell means", show_label=False)
                    with gr.Column():
                        plot_heading("Relative induced correction ΔJ_induced/J_MM · shell means")
                        dressed_relative_realspace_plot = gr.Plot(label="Relative induced correction ΔJ_induced/J_MM · shell means", show_label=False)
                dressed_table = gr.Dataframe(headers=["shell", "radius", "count", "mean J_MM(r)", "mean ΔJ_induced(r)", "mean J_eff(r)", "max |Im J_eff|", "mean J_eff(r)/m_R"], row_count=1, label="Auxiliary-mesh reconstructed robust-space exchange shells", interactive=False)

            with gr.Tab("5 · Magnons"):
                gr.Markdown("### FM-compatible spectra\nIf Gamma is not locally stable, the plot is explicitly signed harmonic data, not a stable magnon spectrum.")
                magnon_info = gr.Markdown()
                plot_heading("Magnon bands")
                magnon_plot = gr.Plot(label="Magnon bands", show_label=False)

            with gr.Tab("6 · Compare datasets"):
                gr.Markdown("### Compare two compatible Jij datasets\nUseful for LKAG vs frozen-magnon comparisons. Basis cell/positions must match; exchange values are allowed to differ.")
                with gr.Row():
                    b_inpsd = gr.File(label="Dataset B · inpsd.dat", file_count="single", type="filepath", elem_classes=["imx-file-upload"])
                    b_files = gr.File(label="Dataset B · referenced files", file_count="multiple", type="filepath", elem_classes=["imx-file-upload"])
                    external_response_file = gr.File(label="Optional external induced response (qx qy qz m_ind or path_coordinate m_ind)", file_count="single", type="filepath", elem_classes=["imx-file-upload"])
                    compare_observable = gr.Dropdown(["raw", "dressed", "magnons"], value="raw", label="Compare observable")
                    compare_button = gr.Button("Compare A vs B", variant="primary")
                compare_status = gr.Markdown()
                compare_table = gr.Dataframe(headers=["check / diagnostic", "value"], row_count=1, label="Compatibility and diagnostics", interactive=False)
                plot_heading("Raw exchange comparison")
                compare_plot = gr.Plot(label="Raw exchange comparison", show_label=False)
                plot_heading("Optional induced-response comparison")
                compare_response_plot = gr.Plot(label="Optional induced-response comparison", show_label=False)

            with gr.Tab("7 · Methods / limitations"):
                gr.Markdown("""
                ### Methods in one page

                **Raw rigid-site model.** Every supplied magnetic site is treated as an ordinary rigid moment. This is diagnostic when some sites are physically induced.

                **Polesya-like induced moments.** Induced moments are eliminated as algebraic variables. The historical mode uses an unweighted local neighbour sum; the selected J-weighted mode evaluates the labelled **J-weighted induced-response approximation**.

                **Mryasov-like downfolding.** The induced subspace is eliminated variationally from one quadratic energy functional, yielding a robust-space `J_eff(q)`. No induced magnon branches are added.

                **Reciprocal sampling.** The exchange, response, dressed-exchange, and magnon plots share a seekpath high-symmetry path. The dressed real-space channel, absolute-correction, and relative-correction plots are reconstructed from a separate complete auxiliary mesh and are not inferred from that line.

                **Important limitation.** Conventional input Jij are not, by themselves, an exact first-principles susceptibility or induction kernel. Results depend on the explicit classification, X values, response mode, and conditioning of `I − X K_II`.

                The conceptual foundations are discussed in Mryasov *et al.* on induced moments and two-ion anisotropy in FePt, and Polesya *et al.* on induced moments in FePd/CoPt. See the repository README for the scope and citations. Unsupported cases are flagged rather than silently converted: noncollinear order, AF/ferrimagnetic LSWT, DMI, exchange tensors, SOC, anisotropy, and frequency-dependent susceptibility.
                """)

        gr.Markdown("### Downloads\nEvery successful analysis writes canonical model JSON, validation diagnostics, raw/dressed J(q), response data, dressed Jij where resolvable, and magnon bands.")
        downloads = gr.File(label="Analysis files", file_count="multiple", interactive=False)

        def do_mapping(inpsd, files):
            try:
                mapping = inspect_upload_mapping(inpsd, files)
                choices = list(mapping.choices)
                updates = [gr.Dropdown(choices=choices, value=mapping.references.get(key)) for key in ("posfile", "momfile", "exchange")]
                return mapping.rows(), *updates
            except Exception as exc:
                return [["error", str(exc), "unavailable"]], gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

        def _induced_site_update(model):
            default_sites = default_induced_sites(model)
            choices = [
                (f"site {site.index} · type {site.atom_type} · m={site.moment:.6g} μB", site.index)
                for site in model.sites
                if site.moment is not None
            ]
            return gr.CheckboxGroup(choices=choices, value=list(default_sites))

        def _role_preview(induced_selection, current):
            if current is None:
                return []
            loaded = current.loaded if isinstance(current, AnalysisSession) else current
            if not isinstance(loaded, LoadedUppASD):
                return []
            classification_obj = classification_from_induced_sites(induced_selection, loaded.model)
            return classification_rows(loaded.model, classification_obj.site_role)

        def do_load(current, inpsd, files, selected_example, energy, symmetry_mode_flag, manual_pos_name, manual_mom_name, manual_exchange_name):
            old_session = current if isinstance(current, AnalysisSession) else None
            try:
                expand_symmetry = symmetry_mode_flag == "reduced"
                if selected_example in example_paths:
                    example_dir = example_paths[selected_example].parent
                    example_files = [example_dir / name for name in ("posfile", "momfile", "jfile")]
                    loaded, mapping = load_uploaded_set(example_paths[selected_example], example_files, energy_unit=energy, expand_symmetry=expand_symmetry)
                else:
                    loaded, mapping = load_uploaded_set(inpsd, files, manual_mapping={"posfile": manual_pos_name or "", "momfile": manual_mom_name or "", "exchange": manual_exchange_name or ""}, energy_unit=energy, expand_symmetry=expand_symmetry)
                cleanup_analysis_artifacts(old_session)
                return loaded, f"✅ Loaded `{mapping.inpsd}` with {len(loaded.model.sites)} basis sites and {len(loaded.model.exchange_bonds)} exchange rows.", model_summary(loaded), classification_rows(loaded.model), _induced_site_update(loaded.model), mapping.rows()
            except UploadMappingError as exc:
                return None, f"⚠️ **Could not load input:** {exc}", "", [], gr.CheckboxGroup(choices=[], value=[]), exc.mapping.rows()
            except (FileNotFoundError, InputFormatError, ValueError) as exc:
                return None, f"⚠️ **Could not load input:** {exc}", "", [], gr.CheckboxGroup(choices=[], value=[]), []

        def do_analyse(current, induced_selection, mesh, mode, x_value, include):
            if current is None:
                return (None, "⚠️ Load an input set first.", "", [], None, None, "", [], None, "", None, None, None, None, [], "", None, [])
            old_session = current if isinstance(current, AnalysisSession) else None
            try:
                # The load callback stores LoadedUppASD in gr.State; after the
                # first analysis the same state contains AnalysisSession.
                # Normalize both states before reading the model.  In
                # particular, LoadedUppASD delegates unknown attributes to its
                # MagneticCrystal, so current.loaded.model gives a misleading
                # ``MagneticCrystal has no attribute loaded`` error.
                loaded = current.loaded if isinstance(current, AnalysisSession) else current
                if not isinstance(loaded, LoadedUppASD):
                    raise ValueError("analysis state is neither a loaded UppASD input nor an analysis session")
                classification_obj = classification_from_induced_sites(induced_selection, loaded.model)
                analysis = analyse_model(loaded, classification_obj, mesh_size=int(mesh), mode=mode, x=x_value, include_induced_induced=include)
                cleanup_analysis_artifacts(old_session)
                dressed_rows = []
                if analysis.downfolding is not None:
                    dressed_rows = dressed_shell_rows(analysis)
                delta_max = 0.0 if analysis.downfolding is None else float(np.max(np.abs(analysis.downfolding.delta_induced), initial=0.0))
                path_source = "unknown" if analysis.path is None else analysis.path.path.source
                realspace_note = "controlled auxiliary complete-mesh J_eff(r) is available" if analysis.dressed_real_space is not None else "controlled auxiliary J_eff(r) reconstruction is unavailable"
                rescale_note = "For one positive-moment robust site, the real-space panel also shows J_eff(r)/m_R, the q-dependent exchange scale entering the moment-normalized FM dynamical matrix." if len(analysis.robust_sites) == 1 else "The robust-space result remains matrix-valued for multiple robust sites; no scalar rescaling is implied."
                dressed_text = ordering_markdown("Robust-only", analysis.robust_ordering) + "  \n" + ordering_markdown("Dressed", analysis.dressed_ordering) + f"  \n**Induced correction:** max |ΔJ(q)| = `{delta_max:.6g}`  \n**q sampling:** seekpath high-symmetry path (`{path_source}`)  \n**Real-space output:** {realspace_note}; it is not obtained by inverse-transforming the 1-D path.  \n{rescale_note}"
                return (analysis, "✅ Analysis complete.", ordering_markdown("Raw", analysis.raw_ordering), shell_rows(analysis.loaded.model), _figure_exchange(analysis), _figure_realspace(analysis), response_markdown(analysis), inference_rows(analysis.inference), _figure_response(analysis), dressed_text, _figure_dressed(analysis), _figure_dressed_realspace(analysis), _figure_dressed_realspace_delta(analysis), _figure_dressed_realspace_relative(analysis), dressed_rows, magnon_markdown(analysis), _figure_magnons(analysis), export_files(analysis))
            except Exception as exc:
                return (current, f"⚠️ Analysis failed: {exc}", "", [], None, None, f"⚠️ {exc}", [], None, "", None, None, None, None, [], f"⚠️ {exc}", None, [])

        def do_compare(current, b_input, b_supporting, external_response, observable, energy, symmetry_mode_flag, induced_selection, mesh, mode, x_value, include):
            if current is None:
                return "⚠️ Load and analyze Dataset A first.", [], None, None
            b_analysis = None
            try:
                b_loaded, _ = load_uploaded_set(b_input, b_supporting, energy_unit=energy, expand_symmetry=symmetry_mode_flag == "reduced")
                classification_obj = classification_from_induced_sites(induced_selection, current.loaded.model)
                b_classification = classification_from_induced_sites(induced_selection, b_loaded.model)
                b_analysis = analyse_model(b_loaded, b_classification, mesh_size=int(mesh), mode=mode, x=x_value, include_induced_induced=include)
                from induced_exchange import ExchangeDataset, compare_exchange_datasets
                a_dataset = ExchangeDataset(current.loaded.model, label="Dataset A", robust_sites=classification_obj.robust_sites, induced_sites=classification_obj.induced_sites, x=_parse_x_for_compare(x_value), mode=mode)
                b_dataset = ExchangeDataset(b_analysis.loaded.model, label="Dataset B", robust_sites=b_analysis.robust_sites, induced_sites=b_analysis.induced_sites, x=_parse_x_for_compare(x_value), mode=mode)
                comparison = compare_exchange_datasets(a_dataset, b_dataset, current.q_fractional, include_magnons=True, stiffness_q_max=default_stiffness_q_max(current.loaded.model), external_response=external_response)
                rows_out = [["compatible", comparison.compatibility.compatible], *[["warning", item] for item in comparison.compatibility.warnings], *[["diagnostic", item] for item in comparison.diagnostics]]
                for analysis in (comparison.dataset_a, comparison.dataset_b):
                    for name, stiffness in (('raw', analysis.raw_stiffness), ('robust-only', analysis.robust_stiffness), ('dressed', analysis.dressed_stiffness)):
                        if stiffness is not None:
                            rows_out.append([f"{analysis.dataset.label} {name} stiffness", stiffness.coefficient if stiffness.coefficient is not None else "unavailable"])
                response_plot = None
                if comparison.induced_response is not None:
                    induced = comparison.induced_response
                    if induced.external is not None:
                        rows_out.append(["external induced response source", induced.external.source or "uploaded data"])
                        for label, metrics in (("Dataset A", induced.metrics_a), ("Dataset B", induced.metrics_b)):
                            if metrics is not None:
                                rows_out.append([f"{label} external response relative RMSE", metrics.relative_rmse if metrics.relative_rmse is not None else "unavailable"])
                        from induced_exchange import plot_induced_response_comparison
                        response_plot = plot_induced_response_comparison(comparison)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8.4, 4.1))
                from induced_exchange import plot_comparison
                plot_comparison(comparison, observable=observable, ax=ax)
                ax.set_title(f"Dataset A vs B · {observable}")
                fig.tight_layout()
                if response_plot is not None:
                    response_plot = response_plot.figure
                    plt.close(response_plot)
                plt.close(fig)
                cleanup_analysis_artifacts(b_analysis)
                return f"✅ Compared `{comparison.dataset_a.dataset.label}` and `{comparison.dataset_b.dataset.label}`.", rows_out, fig, response_plot
            except Exception as exc:
                cleanup_analysis_artifacts(b_analysis)
                return f"⚠️ Comparison unavailable: {exc}", [], None, None

        def _parse_x_for_compare(value):
            if value is None or not str(value).strip():
                return None
            text = str(value).strip()
            if ":" in text or "=" in text:
                return {int(k.strip()): float(v) for k, v in (token.replace("=", ":", 1).split(":", 1) for token in text.replace(";", ",").split(",") if token.strip())}
            vals = [float(token.strip()) for token in text.replace(";", ",").split(",") if token.strip()]
            return vals[0] if len(vals) == 1 else vals

        refresh_mapping.click(do_mapping, [inpsd_file, supporting_files], [mapping_table, manual_pos, manual_mom, manual_exchange])
        load_button.click(do_load, [session_state, inpsd_file, supporting_files, example, energy_unit, symmetry_mode, manual_pos, manual_mom, manual_exchange], [session_state, load_status, parsed_summary, classification, induced_site_toggle, mapping_table])
        induced_site_toggle.change(_role_preview, [induced_site_toggle, session_state], classification)
        outputs = [session_state, load_status, raw_ordering, shell_table, raw_plot, distance_plot, response_info, inference_table, response_plot, dressed_info, dressed_plot, dressed_realspace_plot, dressed_delta_realspace_plot, dressed_relative_realspace_plot, dressed_table, magnon_info, magnon_plot, downloads]
        for button in (analyze_button, response_recompute):
            button.click(do_analyse, [session_state, induced_site_toggle, mesh_size, response_mode, x_override, include_induced], outputs)
        compare_button.click(do_compare, [session_state, b_inpsd, b_files, external_response_file, compare_observable, energy_unit, symmetry_mode, induced_site_toggle, mesh_size, response_mode, x_override, include_induced], [compare_status, compare_table, compare_plot, compare_response_plot])

    return demo


if __name__ == "__main__":
    import gradio as gr

    demo = build_demo()
    demo.launch(**_configure_launch_options(demo, gr))
