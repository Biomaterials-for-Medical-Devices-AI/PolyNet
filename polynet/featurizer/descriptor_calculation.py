from rdkit.Chem import Descriptors, MolFromSmiles


def calculate_descriptors(smiles_list: list) -> tuple[dict, list]:
    """
    Calculate molecular descriptors for a list of SMILES strings.

    Args:
        smiles_list: list of SMILES strings

    Returns:
        dict: dictionary of SMILES strings and their corresponding descriptors


    """

    descriptors = {}
    descriptors_names = [x[0] for x in Descriptors.descList]

    for smiles in smiles_list:
        m = MolFromSmiles(smiles)

        if m is None:
            print(f"Error in processing molecule {smiles}")
            continue

        descriptors[smiles] = [func(m) for _, func in Descriptors.descList]

    return descriptors, descriptors_names
