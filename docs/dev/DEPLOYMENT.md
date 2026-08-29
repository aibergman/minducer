# Development and deployment guide

## Runtime profile

The repository is ready for a standard CPU Hugging Face Gradio Space. The
Space configuration is the YAML block at the top of `README.md`; it selects
`app.py`, Python 3.10, and the pinned Gradio version. Runtime packages are in
`requirements.txt`:

```text
gradio==5.50.0
matplotlib>=3.7,<4
numpy>=1.23,<3
seekpath>=2.1,<3
spglib>=2.0,<3
```

`app.py` adds the local `src/` directory to the import path, so a source-layout
checkout runs directly without a separate package-install step.

## Local development

Create an environment, install the application dependencies, and run the test
suite from the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src pytest -q
python app.py
```

Open the local Gradio URL and load a bundled example before changing analysis
behaviour. The package should also remain usable without Gradio:

```bash
python -m pip install -e .
induced-exchange-uppasd examples/fept_style/inpsd.dat
```

Keep scientific calculations in `src/induced_exchange/`; `app.py` and
`src/induced_exchange/space.py` should remain orchestration and presentation
layers. Add or update tests with any behavioural change, especially one that
affects input conventions, units, Fourier phases, conditioning diagnostics, or
the magnetic model.

## Deploy to Hugging Face Spaces

1. Create a Hugging Face Space with the **Gradio** SDK and a CPU hardware tier
   appropriate for the intended datasets.
2. Push this repository's release contents to the Space repository. Keep the
   README YAML metadata, `app.py`, `requirements.txt`, `src/`, and any examples
   you want visible in the deployed app.
3. Wait for the Space build to complete, then open its public URL. Hugging Face
   provides the public endpoint, so the application does not use `share=True`.
4. Verify an upload containing separate `inpsd.dat`, `posfile`, `momfile`, and
   exchange files, as well as a bundled example.

## Operational notes

- The application keeps analysis artifacts in temporary storage and cleans
  prior session artifacts when a new input is loaded. Do not treat Space local
  disk as durable user storage.
- Uploaded inputs are scientific data. Set Space visibility, access controls,
  and repository history according to the sensitivity of the material.
- The interactive mesh selector is limited to CPU-friendly resolutions. Users
  needing large reciprocal meshes should use the library in their own compute
  environment.
- `spglib` expansion is opt-in in the UI. It should be enabled only for
  symmetry-reduced exchange files, not complete neighbour lists.
- Pinning Gradio matters because the app contains a small compatibility layer
  for Gradio's 5-to-6 launch/theme API transition. Test a deliberate dependency
  upgrade locally before publishing it.

## Release checklist

- [ ] `PYTHONPATH=src pytest -q` passes.
- [ ] `python app.py` starts without import or Gradio warnings that affect use.
- [ ] A bundled example completes through download creation.
- [ ] README metadata and `requirements.txt` match the intended runtime.
- [ ] No private input data, credentials, AI prompts, or local build artifacts
      are included in the commit.
