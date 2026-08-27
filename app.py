"""Hugging Face / Gradio entry point for the Induced-Moment Exchange Explorer."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# The Space runs ``python app.py`` directly from a source-layout repository.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Gradio imports a plotting component during startup.  Keep Matplotlib's
# cache in a writable ephemeral location on minimal Space containers.
MPL_CACHE = Path(tempfile.gettempdir()) / "induced_exchange_mpl"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))


def build_demo():
    """Build the UI lazily so the scientific package remains Gradio-free."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised on bare library installs
        raise RuntimeError("The Space UI needs Gradio. Install the dependencies from requirements.txt.") from exc

    from induced_exchange.space import (
        UploadMappingError,
        _figure_dressed,
        _figure_exchange,
        _figure_magnons,
        _figure_path,
        _figure_realspace,
        _figure_response,
        analyse_model,
        classification_rows,
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
        _classification_from_rows,
    )

    css = """
    :root { --imx-ink: #183b56; --imx-teal: #18a999; --imx-coral: #e07a5f; --imx-paper: #f7fafc; }
    body { background: #eef3f6; }
    .imx-shell { max-width: 1380px; margin: 0 auto; }
    .imx-hero { background: linear-gradient(125deg, #183b56 0%, #236477 58%, #18a999 100%); color: white; border-radius: 20px; padding: 30px 34px; margin-bottom: 18px; box-shadow: 0 14px 32px rgba(24,59,86,.16); }
    .imx-hero h1 { margin: 0 0 8px; font-size: 2.2rem; letter-spacing: -.03em; }
    .imx-hero p { margin: 0; max-width: 860px; color: #e7f4f2; font-size: 1.05rem; }
    .imx-kicker { text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 700; color: #a9ebe0; }
    .imx-note { border-left: 4px solid var(--imx-coral); background: #fff7f3; padding: 12px 15px; border-radius: 8px; color: #5b3b35; }
    .imx-card { border: 1px solid #dfe8ed; background: rgba(255,255,255,.72); border-radius: 14px; padding: 6px; }
    .imx-muted { color: #597080; }
    footer { display: none !important; }
    """

    examples_root = Path(__file__).resolve().parent / "examples"
    example_choices = ["Upload my files", "Simple analytic FM", "Induced-moment toy", "FePt-style UppASD"]
    example_paths = {
        "Simple analytic FM": examples_root / "simple_fm" / "inpsd.dat",
        "Induced-moment toy": examples_root / "induced_toy" / "inpsd.dat",
        "FePt-style UppASD": examples_root / "fept_style" / "inpsd.dat",
    }

    with gr.Blocks(title="Induced-Moment Exchange Explorer", theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"), css=css) as demo:
        session_state = gr.State(None)
        gr.HTML("<div class='imx-hero'><div class='imx-kicker'>Scientific exchange analysis · CPU-friendly</div><h1>Induced-Moment Exchange Explorer</h1><p>From UppASD-style Jij input to reciprocal exchange, induced response, downfolded interactions, and FM-compatible magnon diagnostics.</p></div>")

        with gr.Tabs():
            with gr.Tab("1 · Input"):
                gr.Markdown("### Bring an UppASD input set into the explorer\nUpload `inpsd.dat` and its referenced files, or leave it blank and upload `posfile`, `momfile`, and `jfile` individually. References are matched by basename because browser uploads do not preserve arbitrary relative paths; manual assignment is available below.")
                with gr.Row():
                    example = gr.Dropdown(example_choices, value="Upload my files", label="Starting point")
                    energy_unit = gr.Dropdown(["unspecified", "meV", "mRy", "eV"], value="unspecified", label="Energy unit (confirm input)")
                    length_unit = gr.Textbox(value="unspecified", label="Length unit", scale=1)
                with gr.Row():
                    inpsd_file = gr.File(label="inpsd.dat", file_count="single", type="filepath")
                    supporting_files = gr.File(label="posfile / momfile / jfile", file_count="multiple", type="filepath")
                with gr.Row():
                    refresh_mapping = gr.Button("Inspect file mapping", variant="secondary")
                    load_button = gr.Button("Load input set", variant="primary")
                mapping_table = gr.Dataframe(headers=["reference", "uploaded basename", "status"], datatype=["str", "str", "str"], label="Referenced-file mapping", interactive=False)
                with gr.Row():
                    manual_pos = gr.Dropdown([], label="Manual posfile assignment", allow_custom_value=False)
                    manual_mom = gr.Dropdown([], label="Manual momfile assignment", allow_custom_value=False)
                    manual_exchange = gr.Dropdown([], label="Manual exchange assignment", allow_custom_value=False)
                load_status = gr.Markdown()
                parsed_summary = gr.Markdown()
                classification = gr.Dataframe(headers=["site", "atom_type", "moment (mu_B)", "role"], datatype=["number", "str", "number", "str"], label="Explicit site classification (edit role: robust / induced)", interactive=True)
                with gr.Row():
                    mesh_size = gr.Slider(4, 16, value=8, step=1, label="Reciprocal mesh per axis", info="CPU default: 8³ points; larger meshes increase work.")
                    analyze_button = gr.Button("Run analysis", variant="primary")

            with gr.Tab("2 · Raw exchange"):
                gr.Markdown("### Raw rigid-site exchange\nAll supplied sites and Jij rows are retained. The ordering diagnostic is the largest eigenvalue of J(q) under `H = −1/2 Σ Jij eᵢ·eⱼ`.")
                raw_ordering = gr.Markdown("Load an input set and run analysis.")
                with gr.Row():
                    raw_plot = gr.Plot(label="J(q) eigenvalue scan")
                    distance_plot = gr.Plot(label="Jij versus distance")
                with gr.Row():
                    path_plot = gr.Plot(label="High-symmetry path")
                shell_table = gr.Dataframe(headers=["shell", "radius", "count", "mean Jij", "min Jij", "max Jij"], label="Radial exchange shells", interactive=False)

            with gr.Tab("3 · Induced response"):
                gr.Markdown("### Induced/slave response\nClassify sites explicitly in the Input tab. The induced response is algebraic and instantaneous; induced sites are not independent LLG/LSWT degrees of freedom.")
                with gr.Row():
                    response_mode = gr.Radio([("J-weighted approximation", "j_weighted"), ("Historical unweighted", "historical")], value="j_weighted", label="Response model")
                    x_override = gr.Textbox(label="X override", placeholder="blank = infer · e.g. 0.12 or 2:0.12, 3:0.08", info="X is susceptibility-like, not m_ind.")
                    include_induced = gr.Checkbox(value=True, label="Include induced–induced propagation")
                response_recompute = gr.Button("Recompute response and dressing", variant="primary")
                response_info = gr.Markdown()
                with gr.Row():
                    response_plot = gr.Plot(label="m_ind(q) / m_ind(0)")
                    inference_table = gr.Dataframe(headers=["site", "m_ref", "source field", "X", "warnings"], label="X inference and source-field normalization", interactive=False)

            with gr.Tab("4 · Dressed exchange"):
                gr.Markdown("### Robust-space exchange after induced-moment elimination\nThe induced contribution is a model-dependent correction using the selected response approximation.")
                dressed_info = gr.Markdown()
                dressed_plot = gr.Plot(label="Raw robust-only vs dressed J_eff(q)")
                dressed_table = gr.Dataframe(headers=["shell", "radius", "count", "mean Jij", "min Jij", "max Jij"], label="Finite-q reconstructed dressed Jij shells", interactive=False)

            with gr.Tab("5 · Magnons"):
                gr.Markdown("### FM-compatible spectra\nIf Gamma is not locally stable, the plot is explicitly signed harmonic data, not a stable magnon spectrum.")
                magnon_info = gr.Markdown()
                magnon_plot = gr.Plot(label="Magnon bands")

            with gr.Tab("6 · Compare datasets"):
                gr.Markdown("### Compare two compatible Jij datasets\nUseful for LKAG vs frozen-magnon comparisons. Basis cell/positions must match; exchange values are allowed to differ.")
                with gr.Row():
                    b_inpsd = gr.File(label="Dataset B · inpsd.dat", file_count="single", type="filepath")
                    b_files = gr.File(label="Dataset B · referenced files", file_count="multiple", type="filepath")
                    compare_observable = gr.Dropdown(["raw", "dressed", "magnons"], value="raw", label="Compare observable")
                    compare_button = gr.Button("Compare A vs B", variant="primary")
                compare_status = gr.Markdown()
                compare_table = gr.Dataframe(headers=["check / diagnostic", "value"], label="Compatibility and diagnostics", interactive=False)
                compare_plot = gr.Plot(label="Raw exchange comparison")

            with gr.Tab("7 · Methods / limitations"):
                gr.Markdown("""
                ### Methods in one page

                **Raw rigid-site model.** Every supplied magnetic site is treated as an ordinary rigid moment. This is diagnostic when some sites are physically induced.

                **Polesya-like slave moments.** Induced moments are eliminated as algebraic variables. The historical mode uses an unweighted local neighbour sum; the selected J-weighted mode evaluates the labelled **J-weighted induced-response approximation**.

                **Mryasov-like downfolding.** The induced subspace is eliminated variationally from one quadratic energy functional, yielding a robust-space `J_eff(q)`. No induced magnon branches are added.

                **Important limitation.** Conventional input Jij are not, by themselves, an exact first-principles susceptibility or induction kernel. Results depend on the explicit classification, X values, response mode, and conditioning of `I − X K_mm`.

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

        def do_load(inpsd, files, selected_example, energy, length, manual_pos_name, manual_mom_name, manual_exchange_name):
            try:
                if selected_example in example_paths:
                    example_dir = example_paths[selected_example].parent
                    example_files = [example_dir / name for name in ("posfile", "momfile", "jfile")]
                    loaded, mapping = load_uploaded_set(example_paths[selected_example], example_files, energy_unit=energy, length_unit=length)
                else:
                    loaded, mapping = load_uploaded_set(inpsd, files, manual_mapping={"posfile": manual_pos_name or "", "momfile": manual_mom_name or "", "exchange": manual_exchange_name or ""}, energy_unit=energy, length_unit=length)
                return loaded, f"✅ Loaded `{mapping.inpsd}` with {len(loaded.model.sites)} basis sites and {len(loaded.model.exchange_bonds)} exchange rows.", model_summary(loaded), classification_rows(loaded.model), mapping.rows()
            except UploadMappingError as exc:
                return None, f"⚠️ **Unresolved upload references:** {exc}", "", [], exc.mapping.rows()
            except (FileNotFoundError, InputFormatError, ValueError) as exc:
                return None, f"⚠️ **Could not load input:** {exc}", "", [], []

        def do_analyse(current, rows, mesh, mode, x_value, include):
            if current is None:
                return (None, "⚠️ Load an input set first.", "", [], None, None, None, "", [], None, "", None, [], "", None, [])
            try:
                classification_obj = _classification_from_rows(rows, current.model)
                analysis = analyse_model(current, classification_obj, mesh_size=int(mesh), mode=mode, x=x_value, include_induced_induced=include)
                dressed_rows = []
                if analysis.downfolding is not None:
                    dressed_rows = dressed_shell_rows(analysis)
                delta_max = 0.0 if analysis.downfolding is None else float(np.max(np.abs(analysis.downfolding.delta_induced), initial=0.0))
                dressed_text = ordering_markdown("Robust-only", analysis.robust_ordering) + "  \n" + ordering_markdown("Dressed", analysis.dressed_ordering) + f"  \n**Induced correction:** max |ΔJ(q)| = `{delta_max:.6g}`"
                return (analysis, "✅ Analysis complete.", ordering_markdown("Raw", analysis.raw_ordering), shell_rows(analysis.loaded.model), _figure_exchange(analysis), _figure_realspace(analysis), _figure_path(analysis), response_markdown(analysis), inference_rows(analysis.inference), _figure_response(analysis), dressed_text, _figure_dressed(analysis), dressed_rows, magnon_markdown(analysis), _figure_magnons(analysis), export_files(analysis))
            except Exception as exc:
                return (current, f"⚠️ Analysis failed: {exc}", "", [], None, None, None, f"⚠️ {exc}", [], None, "", None, [], f"⚠️ {exc}", None, [])

        def do_compare(current, b_input, b_supporting, observable, energy, rows, mesh, mode, x_value, include):
            if current is None:
                return "⚠️ Load and analyze Dataset A first.", [], None
            try:
                b_loaded, _ = load_uploaded_set(b_input, b_supporting, energy_unit=energy)
                classification_obj = _classification_from_rows(rows, current.model)
                b_classification = _classification_from_rows(rows, b_loaded.model)
                b_analysis = analyse_model(b_loaded, b_classification, mesh_size=int(mesh), mode=mode, x=x_value, include_induced_induced=include)
                from induced_exchange import ExchangeDataset, compare_exchange_datasets
                a_dataset = ExchangeDataset(current.loaded.model, label="Dataset A", robust_sites=current.robust_sites, induced_sites=current.induced_sites, x=_parse_x_for_compare(x_value), mode=mode)
                b_dataset = ExchangeDataset(b_analysis.loaded.model, label="Dataset B", robust_sites=b_analysis.robust_sites, induced_sites=b_analysis.induced_sites, x=_parse_x_for_compare(x_value), mode=mode)
                comparison = compare_exchange_datasets(a_dataset, b_dataset, current.q_fractional, include_magnons=True, stiffness_q_max=0.1)
                rows_out = [["compatible", comparison.compatibility.compatible], *[["warning", item] for item in comparison.compatibility.warnings], *[["diagnostic", item] for item in comparison.diagnostics]]
                for analysis in (comparison.dataset_a, comparison.dataset_b):
                    for name, stiffness in (('raw', analysis.raw_stiffness), ('robust-only', analysis.robust_stiffness), ('dressed', analysis.dressed_stiffness)):
                        if stiffness is not None:
                            rows_out.append([f"{analysis.dataset.label} {name} stiffness", stiffness.coefficient if stiffness.coefficient is not None else "unavailable"])
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8.4, 4.1))
                from induced_exchange import plot_comparison
                plot_comparison(comparison, observable=observable, ax=ax)
                ax.set_title(f"Dataset A vs B · {observable}")
                fig.tight_layout()
                return f"✅ Compared `{comparison.dataset_a.dataset.label}` and `{comparison.dataset_b.dataset.label}`.", rows_out, fig
            except Exception as exc:
                return f"⚠️ Comparison unavailable: {exc}", [], None

        def _parse_x_for_compare(value):
            if value is None or not str(value).strip():
                return None
            text = str(value).strip()
            if ":" in text or "=" in text:
                return {int(k.strip()): float(v) for k, v in (token.replace("=", ":", 1).split(":", 1) for token in text.replace(";", ",").split(",") if token.strip())}
            vals = [float(token.strip()) for token in text.replace(";", ",").split(",") if token.strip()]
            return vals[0] if len(vals) == 1 else vals

        refresh_mapping.click(do_mapping, [inpsd_file, supporting_files], [mapping_table, manual_pos, manual_mom, manual_exchange])
        load_button.click(do_load, [inpsd_file, supporting_files, example, energy_unit, length_unit, manual_pos, manual_mom, manual_exchange], [session_state, load_status, parsed_summary, classification, mapping_table])
        outputs = [session_state, load_status, raw_ordering, shell_table, raw_plot, distance_plot, path_plot, response_info, inference_table, response_plot, dressed_info, dressed_plot, dressed_table, magnon_info, magnon_plot, downloads]
        for button in (analyze_button, response_recompute):
            button.click(do_analyse, [session_state, classification, mesh_size, response_mode, x_override, include_induced], outputs)
        compare_button.click(do_compare, [session_state, b_inpsd, b_files, compare_observable, energy_unit, classification, mesh_size, response_mode, x_override, include_induced], [compare_status, compare_table, compare_plot])

    return demo


if __name__ == "__main__":
    build_demo().launch()
