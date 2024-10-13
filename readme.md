# Synthetic Amazon Reviews Generator

A Python project to generate synthetic Amazon reviews for supplements and vitamins using the llama3 model via Ollama.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install Ollama](#install-ollama)
  - [Pull the llama3 Model](#pull-the-llama3-model)
  - [Upgrade pip (Optional)](#upgrade-pip-optional)
  - [Install Python Dependencies](#install-python-dependencies)
- [Usage](#usage)
- [Verification](#verification)
- [License](#license)

## Installation

### Prerequisites

- **Ollama**: Ensure Ollama is installed on your system.
- **Python 3.8+**: Required for running the Python scripts.
- **pip**: Python package manager.

### Install Ollama

Run the following command to install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull the llama3 Model

Download the `llama3` model with:

```bash
ollama pull llama3
```

Verify the installation by listing available models:

```bash
ollama list
```

Expected output:

```
Available models:
- llama3
```

### Upgrade pip (Optional)

It's recommended to upgrade pip to the latest version:

```bash
pip install --upgrade pip
```

### Install Python Dependencies

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Source Data**:

   Ensure you have the `reviews_supplements.csv` file in the project directory. This file should contain a column named `text` with the original reviews.

2. **Run the Script**:

   Execute the main script to generate synthetic reviews:

   ```bash
   python main.py
   ```

3. **Outputs**:

   - `synthetic_reviews.json`: Contains the generated synthetic reviews.
   - `evaluation_metrics.png`: Visualizations comparing synthetic and source data.
   - Console logs: Progress updates and evaluation metrics.

## Verification

To ensure the `llama3` model is correctly installed, run:

```bash
ollama list
```

You should see:

```
Available models:
- llama3
```

## License

[MIT License](LICENSE)
