# Hugging Face Spaces deployment audit

Checked 2026-08-28 for the Gradio Space entry point in this repository.

## Ready for a basic CPU Space

- `app.py` is at the repository root and launches the demo when run directly.
- The README has Hugging Face Space metadata for Gradio, the tested Gradio
  version, Python 3.10, and the root app file.
- `requirements.txt` contains the runtime packages used by the Space. Gradio
  is pinned to the tested 5.50 series; the scientific packages have bounded
  major versions.
- The app has no secrets, external services, GPU requirements, apt packages, or
  `packages.txt` dependency.
- Matplotlib uses an ephemeral writable cache under the system temporary
  directory.
- Browser uploads require `inpsd.dat`, are matched by basename, are limited to
  32 files and 100 MiB total, and cannot write outside the private staging
  directory. Staged input files and generated analysis exports are removed
  after they are no longer needed.
- The Space does not request `share=True`; Hugging Face supplies the deployed
  Space URL.

## Deliberately retained

`pyproject.toml` remains the package-development and CLI metadata source even
though a Space installs its root `requirements.txt`. The examples and
reference files remain in the repository because they are useful for testing
and demonstration.

## Optional slimming before a public release

`examples/uppasd_style/` contains UppASD-generated restart/output artifacts in
addition to the small input/reference files used by the app and tests. They
can be removed or moved to a separate benchmark archive to reduce clone size;
this audit does not delete them. A repository license should also be selected
before making the source public.

## First deployment check

Push the committed repository to a Space configured as a Gradio Space, then
inspect the build logs and open the Input tab. Load both bundled examples and
one uploaded UppASD set. The local verification cannot substitute for the
Space's clean Linux build, so dependency or platform issues should be checked
there before making the Space public.
