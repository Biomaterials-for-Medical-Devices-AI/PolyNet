import streamlit as st

# Set page title and layout
st.set_page_config(page_title="PolyNet", page_icon="🧪", layout="wide")


# Main title
st.title("Welcome to PolyNet!🤩")

# Load and display the logo
st.image("static/logo.png", width=500)

# Description of the app
st.markdown(
    """
**Welcome to PolyNet! 🤩**

This platform enables chemists to **train Graph Neural Networks (GNNs)** on their own molecular datasets to predict properties of polymers (yes, on your own dataset! 🥳). Through this tool, you can generate insights and predictions on polymer properties, even without prior knowledge of machine learning models. The app is designed to be intuitive and chemist-friendly, allowing you to focus on the chemistry while the platform takes care of the computations.

### How to Prepare Your Dataset

To use this platform, your dataset should be organized as a **CSV file** with specific columns that represent both molecular structures and the property you aim to model/predict. Here's how your CSV file should be structured:

1. **SMILES column(s)**: The molecular structures of the monomers must be provided as **SMILES strings**. SMILES is a way to represent a molecule's structure using a line of text, which allows the model to read and process the molecule. Each polymer in your dataset should be placed in a separate row. If your polymer includes multiple molecules (e.g., copolymers), you can use multiple columns for SMILES.

2. **Target property column**: This column represents the property you are trying to predict. It can be a wide variety of molecular properties, such as solubility, toxicity, biodegradability, etc. This value should be included in a separate column and associated with each polymer.

#### Important Note on Molecular Features

While SMILES strings can represent many molecular features, they **cannot fully capture certain structural aspects**, such as **axial chirality**, dispersity, mean molar mass, and other 3D conformations that are critical for some molecular properties. If such features are important for your study, you may need to include additional information in your dataset or preprocess your molecules to ensure these aspects are accurately represented.

For example, features like stereochemistry might need to be explicitly encoded, and this could be done by adding additional columns that detail such molecular characteristics. GNNs rely on how molecules are represented in these datasets, so the completeness and accuracy of this representation are crucial.

#### Example CSV Structure:

| SMILES_1         | SMILES_2        | Mol 2 Axial Chirality | Target_Property |
|------------------|-----------------|-----------------------|-----------------|
| CCO              | CCC             | 1.4                   | 12.5            |
| CCN              | C1=CC=CC=C1     | 1.8                   | 8.4             |
| ...              | ...             | ...                   | ...             |

### Navigation

To make your experience smoother, we’ve organized the platform into three key sections, accessible from the **left side menu**:
1. **👀 Hyperparameter Opt**: This is an optional (but highly recommended) section. Here, you can search for the best hyperparameters to model your specific problem.

2. **👩‍💻 Train GNN**: In this section, you can upload your dataset and configure the settings to train your GNN model (if you didn't do hyperparameter optimization). You will be able to adjust hyperparameters, track progress, and analyze the results.

3. **🫣 Predict**: After training your model, you can use this section to make predictions on new molecules, for which you don't know the property of interest. Simply upload a CSV file with the molecular structures, and the trained GNN will provide predictions for the desired property.

4. **🕵️‍♀️ Explain GNN**: This section offers explainability tools that allow you to understand the rationale behind your GNN’s predictions. Using visualization techniques, you can see which molecular features influence predictions, giving you deeper insights into how the model interprets molecular data.

### Get Started!

Choose a page from the menu to begin training your GNN models or make predictions. We're excited to see how you apply this tool to your molecular studies!
"""
)

# Footer
st.markdown(
    """
---
This application was designed to empower chemists to leverage the power of **Graph Neural Networks** in their molecular studies.

If you come across any issues or have suggestions for improvements, please feel free to reach out to us at: eduardo.aguilar-bejarano@nottingham.ac.uk

If you found our tool useful, please consider citing our work:
"""
)
