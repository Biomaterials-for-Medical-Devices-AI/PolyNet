import streamlit as st

# Set page title and layout
st.set_page_config(page_title="PolyNet", page_icon="🧪")


# Main title
st.title("Welcome to PolyNet!")

# Load and display the logo
st.image("static/logo.png", width=500)

# Description of the app
st.markdown(
    """
**PolyNet** is a no-code platform for training and applying machine learning models to predict polymer properties from molecular structure.
It supports both **Graph Neural Networks (GNNs)**, which learn directly from molecular graphs, and **traditional ML models (TML)** such as Random Forest and Support Vector Machines, which operate on pre-computed molecular descriptors.
You can train either model family independently or both together and compare results.

---

### What PolyNet expects

Your dataset must be a **CSV file** where each row represents one polymer sample. The file must contain:

| Column | Required | Description |
|--------|----------|-------------|
| **SMILES column(s)** | Yes | One column per monomer unit (e.g. `SMILES_1`, `SMILES_2`). Both SMILES and pSMILES notation are supported. |
| **Target property column** | Yes | The property to predict. Use continuous numerical values for **regression** tasks (e.g. glass transition temperature), or integer/string class labels for **classification** tasks (e.g. biodegradable / not biodegradable). |
| **Sample ID column** | Recommended | A unique identifier per row (e.g. `polymer_id`). If omitted, row indices are used. |
| **Weight/fraction columns** | Optional | For copolymers, one column per monomer giving its molar fraction or weight fraction (e.g. `weight_1`, `weight_2`). |
| **Extra descriptor columns** | Optional | Any additional numerical features (e.g. dispersity, mean molar mass) that SMILES alone cannot encode. These can be included as custom descriptors. |

#### Example CSV structure

| id    | SMILES_1        | SMILES_2        | weight_1 | Tg    |
|-------|-----------------|-----------------|----------|-------|
| P001  | CCO             | CCC             | 0.6      | 120.5 |
| P002  | CCN             | c1ccccc1        | 0.4      | 85.2  |
| ...   | ...             | ...             | ...      | ...   |

> **Note:** SMILES strings capture connectivity and atom types but cannot encode certain structural features such as dispersity, mean molar mass, or tacticity. Include these as extra columns if they are relevant to your property of interest.

---

### Workflow

The app is organised as a sequential pipeline. Work through the pages in order:

1. **Create Experiment** — Name your experiment and select your data file. All outputs (models, results, plots) are saved under this experiment.

2. **Representation** — Choose how to represent your molecules. Select molecular descriptor sets for TML models (e.g. Morgan fingerprints, RDKit descriptors) and/or graph node/edge features for GNNs.

3. **Train Models** — Configure your data splitting strategy (random, scaffold, or bootstrap), then train TML models, GNN models, or both. Evaluation metrics and result plots are generated automatically.

4. **Predict** — Upload a new CSV file (same column structure as your training data, target column optional) to obtain predictions from your trained models.

5. **Explain Models** — Visualise which molecular fragments drive GNN predictions using attribution-based explainability methods.

6. **Analyse Results** — Explore and compare model performance across metrics, splits, and model types.

---

### Get Started

Select **Create Experiment** from the left-hand menu to begin.
"""
)

# Footer
st.markdown(
    """
---
If you encounter any issues or have suggestions, please reach out to: eduardo.aguilar-bejarano@nottingham.ac.uk

If you found PolyNet useful in your research, please consider citing our work:
"""
)
