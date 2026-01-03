# Vid2Dataset — Convert videos into image datasets

Tool to extract frames from videos and build image datasets for machine learning.

## What it does
- Extracts frames from video files and saves them as images for dataset creation.
- Provides a simple Streamlit UI (run `app.py`) to preview and export frames.

## Requirements
- Python 3.8+
- (Recommended) create and use a virtual environment

## Quick start
1. Create and activate a venv:

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

2. Install dependencies (if you have a `requirements.txt`):

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

4. Open the URL shown by Streamlit (usually http://localhost:8501) and use the UI to load a video and extract frames.

## Project structure
- `app.py` — main Streamlit application

## Usage for datasets
- Choose a video and select frame rate or interval to export consistent images.
- Optionally provide labels or export directory structure suitable for ML training.

## Contributing
Open an issue or submit a pull request. Keep changes small and focused.

## License
MIT — feel free to reuse and modify.
