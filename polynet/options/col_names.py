from polynet.options.enums import Results, IteratorTypes, SplitTypes


def get_iterator_name(split_type):
    """Get the name of the iterator based on the split type."""
    if split_type == SplitTypes.TrainValTest:
        return IteratorTypes.BootstrapIteration.value
    elif split_type == SplitTypes.CrossValidation:
        return IteratorTypes.Fold.value
    else:
        return IteratorTypes.Iteration.value


def get_true_label_column_name(target_variable_name: str) -> str:
    """Get the true label column name based on the target variable name and model name."""

    return (
        f"{Results.Label.value} {target_variable_name}"
        if target_variable_name
        else Results.Label.value
    )


def get_predicted_label_column_name(target_variable_name: str, model_name: str = None) -> str:
    """Get the predicted label column name based on the target variable name and model name."""
    if model_name and target_variable_name:
        return f"{model_name} {Results.Predicted.value} {target_variable_name}"
    elif target_variable_name:
        return f"{Results.Predicted.value} {target_variable_name}"
    elif model_name:
        return f"{model_name} {Results.Predicted.value}"
    else:
        return f"{Results.Predicted.value}"


def get_score_column_name(
    target_variable_name: str, model_name: str = None, class_num: int = 1
) -> str:
    """Get the score column name based on the target variable name and model name."""
    if model_name and target_variable_name:
        return f"{model_name} {Results.Score.value} {target_variable_name} {class_num}"
    elif target_variable_name:
        return f"{Results.Score.value} {target_variable_name} {class_num}"
    elif model_name:
        return f"{model_name} {Results.Score.value} {class_num}"
    else:
        return Results.Score.value + f" {class_num}"
