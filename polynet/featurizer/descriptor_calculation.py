from rdkit.Chem import Descriptors, MolFromSmiles


def calculate_descriptors(
    smiles_list: list[str], descriptors_list: list[str] | str = "all"
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Calculate molecular descriptors for a list of SMILES strings.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        descriptors_list (list[str] | str): List of descriptor names to calculate or "all" to calculate all.

    Returns:
        tuple: A dictionary mapping each SMILES string to its descriptors,
               and a list of the descriptor names used.
    """
    all_descriptor_funcs = dict(Descriptors.descList)

    # Validate and select descriptor functions
    if descriptors_list == "all":
        selected_descriptors = all_descriptor_funcs
    else:
        if not isinstance(descriptors_list, list):
            raise ValueError("descriptors_list must be a list of strings or 'all'.")

        selected_descriptors = {
            name: all_descriptor_funcs[name]
            for name in descriptors_list
            if name in all_descriptor_funcs
        }

        # Warn if any descriptors were not found
        missing = set(descriptors_list) - set(selected_descriptors)
        if missing:
            print(
                f"Warning: The following descriptors were not found and will be skipped: {missing}"
            )

    descriptors = {}

    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)

        if mol is None:
            print(f"Error processing molecule: {smiles}")
            continue

        descriptors[smiles] = [func(mol) for func in selected_descriptors.values()]

    return descriptors, list(selected_descriptors.keys())
