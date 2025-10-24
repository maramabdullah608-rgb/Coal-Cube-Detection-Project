# Coal Cube Detection Project

Lightweight repository for coal defect detection and quality classification using PyTorch models and a small Flask web app for demo and uploads.

## Repository overview

Root files:
- `coal_classified_defect.ipynb` — Notebook for defect classification exploration and results.
- `project2_coal.ipynb` — Project notebook with experiments.
- `Test_coal_model_via_camera.ipynb` — Notebook demonstrating running model against a camera stream.
- `Test_defect_classification.ipynb` — Notebook for testing defect classification flows.

`coal_detection_app/` — Web demo and supporting assets:
- `app.py` — Flask web app (development/demo entrypoint).
- `app_fixed.py` — Alternative or patched version of the app (use if `app.py` shows issues).
- `requirements.txt` — Python dependencies for the web app and model runtime.
- `models/` — Saved model weights (.pth files):
  - `coal_defect_classification_best.pth`
  - `coal_quality_model.pth`
- `static/` — CSS/JS and other static assets.
- `templates/` — Flask HTML templates (e.g. `index.html`).
- `uploads/` — Uploaded images by the web app (created at runtime).

## Quick start (Windows PowerShell)

Prerequisites:
- Python 3.8+ installed and on PATH.
- (Optional) GPU + CUDA if you plan to run large models — PyTorch should be installed with appropriate CUDA support.

1) Create and activate a virtual environment (from the repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies for the web app:

```powershell
pip install --upgrade pip
pip install -r .\coal_detection_app\requirements.txt
# If PyTorch is not included in requirements.txt, install it manually (choose proper CUDA/cuDNN version):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

3) Run the Flask web app (from repo root or `coal_detection_app`):

```powershell
cd .\coal_detection_app
python app.py
# or, if you prefer the alternate script:
# python app_fixed.py
```

Open http://127.0.0.1:5000 in your browser. Use the upload interface to test images or inspect model outputs.

## Notebooks

Open the `.ipynb` files with Jupyter Notebook or JupyterLab and run cells interactively. Notebooks show example preprocessing, model inference, and evaluation steps. They may require the same Python environment as the web app and access to the `models/` folder.

## Models

Pretrained weights are stored in `coal_detection_app/models/`. These are PyTorch `.pth` files. When loading models in code or notebooks, ensure your model class definition matches the state dict structure.

Example snippet (not included in the repo):

```python
# load model (example)
import torch
from your_model_file import CoalModel  # replace with the actual model class used in the notebooks/app
model = CoalModel()
model.load_state_dict(torch.load('coal_detection_app/models/coal_quality_model.pth', map_location='cpu'))
model.eval()
```

## File upload and storage

The web app stores uploaded images in `coal_detection_app/uploads/`. Make sure the running process has write permissions to that directory.

## Contributing

If you want to contribute:
- Create an issue describing the problem or feature.
- Fork, create a feature branch, and open a pull request with a clear description and test steps.

Small ways to help:
- Add clearer docstrings or inline comments in the notebooks.
- Add a requirements or environment file for the notebooks if they differ from the web app.
- Add a Dockerfile or GitHub Actions workflow for CI.

## Troubleshooting

- If models fail to load, check PyTorch version compatibility and whether the model class definitions match the saved state dict.
- If the web app cannot start due to port conflict, either stop the other service or change the port in `app.py`.
- If using PowerShell and activation is blocked, you may need to run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

## License

This repository is provided as-is. Add a license file (for example, `LICENSE` with MIT text) if you want an explicit license. If you want me to add an MIT license file, tell me and I will add it.

## Author

Repo owner: maramabdullah608-rgb (original author)

---

If you'd like, I can also:
- Add an explicit `LICENSE` file (MIT recommended),
- Add a more detailed `requirements.txt` at the repo root listing the notebook dependencies, or
- Add a short Dockerfile for reproducible deployment.

Tell me which of these (if any) you'd like next.