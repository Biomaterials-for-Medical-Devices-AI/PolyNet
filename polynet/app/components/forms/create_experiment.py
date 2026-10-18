import pandas as pd
import streamlit as st

from polynet.app.components.optional_dependencies import require_psmiles_canonicaliser
from polynet.app.options.file_paths import polynet_experiments_base_dir
from polynet.app.options.state_keys import CreateExperimentStateKeys
from polynet.config.constants import ResultColumn
from polynet.config.enums import DatasetName, ProblemType, StringRepresentation
from polynet.data.creator import (
    BENCHMARK_COLUMNS,
    BenchmarkColumns,
    DatasetCreator,
    held_out_dataset,
    subsample_dataset,
)
from polynet.data.structures import (
    canonicalise_structures,
    detect_string_representation,
    find_invalid_structures,
)
from polynet.plotting.data_analysis import show_continuous_distribution, show_label_distribution
from polynet.utils.validation import find_duplicate_ids


@st.cache_data
def _load_csv(csv_file) -> pd.DataFrame:
    return pd.read_csv(csv_file)


# Benchmarks larger than this are subsampled by default, so that a first
# experiment trains in minutes on a laptop.
DEFAULT_BENCHMARK_SAMPLES = 1000
# Smaller benchmarks use all polymers but this many by default, kept aside as
# an external set.
DEFAULT_SMALL_BENCHMARK_HELD_OUT = 50
# Default number of polymers outside the subset offered for download (all of
# them when fewer are left).
BENCHMARK_HELD_OUT_SAMPLES = 100


@st.cache_data
def _load_benchmark_dataset(dataset_name) -> pd.DataFrame:
    return DatasetCreator(dataset_name).create_dataset()


def _default_benchmark_samples(n_total: int) -> int:
    """1,000 polymers for a large benchmark, otherwise all but 50."""
    if n_total > DEFAULT_BENCHMARK_SAMPLES:
        return DEFAULT_BENCHMARK_SAMPLES
    return max(1, n_total - DEFAULT_SMALL_BENCHMARK_HELD_OUT)


def _out_of_range(value: int, high: int, what: str) -> str | None:
    """Error message when ``value`` is not between 1 and ``high``, else ``None``."""
    if 1 <= value <= high:
        return None
    return f"The number of {what} must be between 1 and {high:,}; {value:,} was entered."


def benchmark_sample_size_widget(df: pd.DataFrame, dataset_name: str = "benchmark") -> pd.DataFrame:
    """
    Ask how many polymers of a benchmark to model and return that subset.

    The subset is random but reproducible (``subsample_dataset``), and the
    experiment saves exactly the selected polymers. Any number of the polymers
    left out (``BENCHMARK_HELD_OUT_SAMPLES`` by default) can be downloaded as a
    CSV to try the Predict page (``held_out_dataset``).

    The number inputs have no browser-side bounds: the browser does not send a
    value outside them, so the page would silently keep using the previous
    number. Out-of-range numbers are reported here instead.
    """
    n_total = len(df)
    default = _default_benchmark_samples(n_total)

    if n_total > DEFAULT_BENCHMARK_SAMPLES:
        st.info(
            f"This dataset has {n_total:,} polymers. Training on all of them, especially the "
            "GNNs and their hyperparameter optimisation, can take hours on a laptop without a "
            f"GPU. To explore PolyNet, start with a random subset of {default:,} polymers; "
            f"set the number to {n_total:,} to use the whole dataset."
        )
    n_samples = int(
        st.number_input(
            f"Number of polymers to use (of {n_total:,})",
            value=default,
            step=100 if n_total > DEFAULT_BENCHMARK_SAMPLES else 10,
            # One key per dataset: a value kept from a larger dataset could exceed
            # this one's maximum.
            key=f"{CreateExperimentStateKeys.BenchmarkSamples}_{dataset_name}",
            help="A random subset, the same every time for the same number (a smaller subset "
            "is contained in a larger one). The experiment saves the selected polymers only.",
        )
    )
    if error := _out_of_range(n_samples, n_total, "polymers to use"):
        st.error(error)
        st.stop()
    # Rows keep their position in the full dataset, so that without an ID column
    # a polymer gets the same number whatever the subset size.
    df = df.reset_index(drop=True)

    n_left = n_total - n_samples
    if n_left == 0:
        st.caption(
            f"All {n_total:,} polymers are used in this experiment, so none are left to try "
            "predictions on polymers the models have never seen. To keep some aside as an "
            "external test set, lower the number of polymers."
        )
    else:
        st.markdown(
            f"**Try predictions on new polymers.** The {n_left:,} polymers left out are from "
            "the same dataset but are **not** used in this experiment, so the models never "
            "see them in training, validation or testing. Download as many as you like below "
            "and, after training, upload the file on the **Predict** page: PolyNet predicts "
            "them and, because the file includes their measured values, compares the "
            "predictions with the measurements and shows whether each polymer is inside the "
            "models' applicability domain. This is how well the models do on polymers they "
            "have never seen."
        )
        n_held_out = int(
            st.number_input(
                f"Number of polymers to download (of {n_left:,} left out)",
                value=min(BENCHMARK_HELD_OUT_SAMPLES, n_left),
                step=10,
                key=f"{CreateExperimentStateKeys.BenchmarkHeldOutSamples}_{dataset_name}",
                help="A random selection of the polymers left out, the same every time for "
                "the same numbers.",
            )
        )
        # Also catches a number chosen before the subset grew. The download does not
        # change the experiment, so only the button is disabled.
        error = _out_of_range(n_held_out, n_left, "polymers to download")
        if error:
            st.error(error)
        held_out = df.iloc[:0] if error else held_out_dataset(df, n_samples, n_held_out)
        st.download_button(
            (
                f"Download {len(held_out):,} polymers not used in this experiment (CSV)"
                if not error
                else "Download polymers not used in this experiment (CSV)"
            ),
            data=held_out.to_csv(index=False),
            file_name=f"{dataset_name}_external_{len(held_out)}.csv",
            mime="text/csv",
            key=f"{CreateExperimentStateKeys.BenchmarkHeldOut}_{dataset_name}",
            disabled=error is not None,
        )

    return subsample_dataset(df, n_samples)


# Thin cached wrappers around the shared, Streamlit-free structure preparation
# in ``polynet.data.structures`` (also used by the CLI and ``predict_external``).


@st.cache_data
def _validate_smiles_cols(df: pd.DataFrame, smiles_cols: tuple, str_representation: str) -> dict:
    return find_invalid_structures(df, list(smiles_cols), str_representation)


@st.cache_data
def _determine_string_representation(df: pd.DataFrame, smiles_cols: tuple) -> str:
    return detect_string_representation(df, list(smiles_cols))


@st.cache_data
def _canonicalise_df(df: pd.DataFrame, smiles_cols: tuple, str_representation: str) -> pd.DataFrame:
    return canonicalise_structures(df, list(smiles_cols), str_representation)


@st.cache_data
def _plot_label_distribution(df: pd.DataFrame, target_col: str, title: str, class_names: tuple):
    return show_label_distribution(
        data=df, target_variable=target_col, title=title, class_names=dict(class_names)
    )


@st.cache_data
def _plot_continuous_distribution(df: pd.DataFrame, target_col: str, title: str):
    return show_continuous_distribution(data=df, target_variable=target_col, title=title)


def _row_numbering(df: pd.DataFrame) -> str:
    """How polymers are numbered when they have no (unique) ID column."""
    if df.index.equals(pd.RangeIndex(len(df))):
        return f"numbered from 0 to {len(df) - 1} in row order"
    return (
        "numbered by their row in the full benchmark dataset (0 is its first row), so each "
        "polymer keeps the same number whatever the subset size"
    )


# Column choices fixed for a built-in benchmark (and cleared when leaving it, as
# the uploaded file will not have these columns).
_LOCKED_BENCHMARK_KEYS = (
    CreateExperimentStateKeys.SmilesCols,
    CreateExperimentStateKeys.IDCol,
    CreateExperimentStateKeys.TargetVariableCol,
    CreateExperimentStateKeys.ProblemType,
)
# Plot labels: defaults for a benchmark, still editable.
_BENCHMARK_LABEL_KEYS = (
    CreateExperimentStateKeys.TargetVariableName,
    CreateExperimentStateKeys.TargetVariableUnits,
)
_BENCHMARK_DEFAULTS_FOR = "benchmark_defaults_for"
# Target column the problem type, target name and class-name inputs belong to,
# so that what was chosen for one target is not kept for another.
_TARGET_DEFAULTS_FOR = "target_defaults_for"


def _reset_target_defaults(
    target_col: str, suggested_problem_type: ProblemType | None, benchmark: BenchmarkColumns | None
) -> None:
    """
    Apply the defaults of a newly selected target column to the form.

    A keyed widget keeps the value of its first run, ignoring ``value`` and
    ``index`` afterwards (Streamlit 1.5 and newer), so the suggested problem
    type, the target name used in the plots and the class names would otherwise
    stay those of the target column selected before. The values are written to
    the session state rather than the keys deleted, so that the widgets also
    *show* them: a deleted key makes the next run fall back to the default
    server-side, but leaves the old value on screen. A benchmark sets its own
    columns and labels (``_sync_benchmark_columns``), which are left alone.
    """
    if st.session_state.get(_TARGET_DEFAULTS_FOR) == target_col:
        return
    st.session_state[_TARGET_DEFAULTS_FOR] = target_col
    prefix = f"{CreateExperimentStateKeys.ClassNames}_"
    for key in [key for key in st.session_state if str(key).startswith(prefix)]:
        del st.session_state[key]
    if benchmark is not None:
        return
    st.session_state[CreateExperimentStateKeys.TargetVariableName] = target_col
    if suggested_problem_type is not None:
        st.session_state[CreateExperimentStateKeys.ProblemType] = suggested_problem_type


def _sync_benchmark_columns(dataset_name: str | None) -> BenchmarkColumns | None:
    """
    Set the column choices of the selected benchmark (``None``: uploaded data).

    Runs before the column widgets: the locked choices are set on every run;
    the plot labels only when the benchmark is first selected, so that edits
    are kept. Leaving a benchmark clears both.
    """
    previous = st.session_state.get(_BENCHMARK_DEFAULTS_FOR)
    columns = BENCHMARK_COLUMNS.get(dataset_name) if dataset_name else None
    if columns is None:
        if previous is not None:
            for key in (*_LOCKED_BENCHMARK_KEYS, *_BENCHMARK_LABEL_KEYS):
                st.session_state.pop(key, None)
            del st.session_state[_BENCHMARK_DEFAULTS_FOR]
            # The labels of the next target column are set again, even when it
            # happens to be the column the benchmark used.
            st.session_state.pop(_TARGET_DEFAULTS_FOR, None)
        return None

    st.session_state[CreateExperimentStateKeys.SmilesCols] = list(columns.smiles_cols)
    st.session_state[CreateExperimentStateKeys.IDCol] = columns.id_col
    st.session_state[CreateExperimentStateKeys.TargetVariableCol] = columns.target_col
    st.session_state[CreateExperimentStateKeys.ProblemType] = columns.problem_type
    if previous != dataset_name:
        st.session_state[CreateExperimentStateKeys.TargetVariableName] = columns.target_name
        st.session_state[CreateExperimentStateKeys.TargetVariableUnits] = columns.target_units
        st.session_state[_BENCHMARK_DEFAULTS_FOR] = dataset_name
    return columns


def select_data_form():

    class_names = {}

    experiment_name = st.text_input(
        "Experiment name",
        placeholder="Enter a name for your experiment",
        key=CreateExperimentStateKeys.ExperimentName,
        help="This name will be used to identify your experiment in the app.",
    )

    exp_path = polynet_experiments_base_dir() / experiment_name

    if experiment_name and exp_path.exists():
        st.error(
            f"Experiment with name '{experiment_name}' already exists. Please choose a different name."
        )
    if not experiment_name:
        st.error("Please provide an experiment name.")

    dataset_source = st.radio(
        "Select dataset source", ["Upload dataset", "Load benchmarking dataset"], horizontal=True
    )

    df = None
    benchmark = None

    if dataset_source == "Upload dataset":
        _sync_benchmark_columns(None)

        csv_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            key=CreateExperimentStateKeys.DatasetName,
            help="Upload a CSV file containing SMILES strings and the target variable.",
        )

        if not csv_file:
            st.warning("Please upload a CSV file to proceed.")
        else:
            df = _load_csv(csv_file)

    elif dataset_source == "Load benchmarking dataset":

        dataset_name = st.selectbox(
            "Select a benchmark dataset to load",
            options=list(DatasetName),
            key=CreateExperimentStateKeys.DatasetNameLoad,
        )

        df = benchmark_sample_size_widget(_load_benchmark_dataset(dataset_name), dataset_name)
        benchmark = _sync_benchmark_columns(dataset_name)
        if benchmark is not None:
            st.info(
                "The SMILES, ID and target columns and the problem type of this benchmark are "
                "set for you and locked. You can still change the target name and units used "
                "in the plots (Extra options)."
            )

    if df is not None and experiment_name != "":
        st.markdown("**Preview Data**")
        if st.checkbox("Show data provided"):
            st.write(df)

        smiles_cols = st.multiselect(
            "Select SMILES columns",
            options=df.columns.tolist(),
            key=CreateExperimentStateKeys.SmilesCols,
            disabled=benchmark is not None,
        )

        if not smiles_cols:
            st.error("Please select at least one SMILES column.")
            st.stop()

        else:
            # Detect first so validation uses the right parser for the representation.
            str_representation = _determine_string_representation(df, tuple(smiles_cols))

            invalid_smiles = _validate_smiles_cols(df, tuple(smiles_cols), str_representation)
            if invalid_smiles:
                for col, smiles in invalid_smiles.items():
                    st.error(
                        f"Invalid SMILES found in column '{col}'. Example: {', '.join(smiles[:10])}"
                    )

                st.error(
                    "To continue with experiment creation, please provide columns with valid SMILES or PSMILES strings."
                )
                st.stop()

            st.write(f"The `{str_representation}` representation has been identified.")
            st.success(f"`{str_representation}` columns checked successfully.")
            st.session_state[CreateExperimentStateKeys.StringRepresentation] = str_representation

            if str_representation == StringRepresentation.PSMILES:
                require_psmiles_canonicaliser()

        if st.checkbox(
            f"Canonicalise `{str_representation}`",
            key=CreateExperimentStateKeys.CanonicaliseSMILES,
            help="To ensure proper featurization, the SMILES and PSMILES are always canonicalised.",
            value=True,
            disabled=True,
        ):
            try:
                df = _canonicalise_df(df, tuple(smiles_cols), str_representation)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            st.success(f"`{str_representation}` columns canonicalized successfully.")

        # Columns that are already spoken for — exclude from downstream selectors.
        smiles_cols_set = set(smiles_cols)

        # If a previously chosen ID/target column was since selected as a SMILES
        # column, clear it so the selectbox doesn't hold an invalid value.
        if st.session_state.get(CreateExperimentStateKeys.IDCol) in smiles_cols_set:
            st.session_state[CreateExperimentStateKeys.IDCol] = None
        if st.session_state.get(CreateExperimentStateKeys.TargetVariableCol) in smiles_cols_set:
            st.session_state[CreateExperimentStateKeys.TargetVariableCol] = None

        remaining_after_smiles = [col for col in df.columns if col not in smiles_cols_set]

        id_col = st.selectbox(
            "Select column with the ID of each molecule",
            options=remaining_after_smiles,
            index=None,
            key=CreateExperimentStateKeys.IDCol,
            disabled=benchmark is not None,
        )

        duplicate_ids = find_duplicate_ids(df[id_col]) if id_col is not None else []
        if duplicate_ids:
            st.warning(
                f"The ID column '{id_col}' has {len(duplicate_ids)} duplicated value(s), e.g. "
                f"{', '.join(map(str, duplicate_ids[:5]))}. IDs must be unique, so the samples "
                f"will be {_row_numbering(df)} instead "
                f"('{id_col}' is kept as a regular column)."
            )
            df.index.name = ResultColumn.INDEX
            df = df.reset_index()
        elif id_col is None:
            st.warning("An ID column to identify your polymers is highly recommended.")
            st.info(f"To differentiate the polymers, they will be {_row_numbering(df)}.")
            df.index.name = ResultColumn.INDEX
            df = df.reset_index()
            # st.session_state[CreateExperimentStateKeys.IDCol] = ResultColumn.INDEX

        # Exclude both SMILES columns and the chosen ID column from the target selector.
        used_cols = smiles_cols_set | ({id_col} if id_col else set())
        if st.session_state.get(CreateExperimentStateKeys.TargetVariableCol) in used_cols:
            st.session_state[CreateExperimentStateKeys.TargetVariableCol] = None

        remaining_after_id = [col for col in df.columns if col not in used_cols]

        target_col = st.selectbox(
            "Select target column",
            options=remaining_after_id,
            index=None,
            key=CreateExperimentStateKeys.TargetVariableCol,
            help="This column should contain the target variable you want to model.",
            disabled=benchmark is not None,
        )

        if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
            st.write(f"Target variable '{target_col}' is numeric")
            unique_vals_target = df[target_col].nunique()
            if unique_vals_target < 20:
                suggested_problem_type = ProblemType.Classification
            else:
                suggested_problem_type = ProblemType.Regression
            if benchmark is None:
                st.warning(f"It looks like a {suggested_problem_type} problem is to be modeled.")
        elif target_col is None:
            st.error("Please select a target variable.")
            return False
        else:
            st.error(
                f"Target variable '{target_col}' is not numeric. Please select a numeric column."
            )
            suggested_problem_type = None
            return False

        if target_col:
            _reset_target_defaults(
                target_col=target_col,
                suggested_problem_type=suggested_problem_type,
                benchmark=benchmark,
            )

            problems = [ProblemType.Classification, ProblemType.Regression]
            problem_type = st.selectbox(
                "Select the problem type",
                problems,
                # The suggested problem type (and a benchmark's) is already in
                # the session state: an index here would only compete with it,
                # and Streamlit warns when a widget has both.
                index=None,
                key=CreateExperimentStateKeys.ProblemType,
                disabled=benchmark is not None,
            )

            if problem_type == ProblemType.Classification:
                n_classes = df[target_col].nunique()
            else:
                n_classes = 1

            # Written to the session state directly: a widget with a key keeps
            # the value of its first run, so a keyed input here would still
            # report the class count of the problem type selected before.
            st.session_state[CreateExperimentStateKeys.NumClasses] = n_classes
            st.text_input(
                "Number of classes",
                value=str(n_classes),
                help="This will be used to create the plots and log information.",
                disabled=True,
            )

            with st.expander("Extra options"):
                target_name = st.text_input(
                    "Target variable name",
                    # Both the benchmark's name and the target column's are put
                    # in the session state before this runs.
                    key=CreateExperimentStateKeys.TargetVariableName,
                    help="This name will be used to create the plots and log information.",
                )

                st.text_input(
                    "Target variable units",
                    key=CreateExperimentStateKeys.TargetVariableUnits,
                    help="This will be used to create the plots and log information.",
                )

                if problem_type == ProblemType.Classification:

                    class_names = {}

                    for vals in sorted(df[target_col].unique()):
                        class_name = st.text_input(
                            f"Class {vals} name",
                            value=str(vals),
                            key=f"{CreateExperimentStateKeys.ClassNames}_{vals}",
                            help="This name will be used to create the plots and log information.",
                        )
                        class_names[str(vals)] = class_name
                    st.session_state[CreateExperimentStateKeys.ClassNames] = class_names

                    st.markdown("**Label Distribution**")

                else:
                    # A regression target has no classes: drop any names left
                    # from a classification target selected earlier.
                    st.session_state.pop(CreateExperimentStateKeys.ClassNames, None)

            if problem_type == ProblemType.Classification:
                title = (
                    f"Label Distribution for {target_name}" if target_name else "Label Distribution"
                )
                fig = _plot_label_distribution(
                    df=df,
                    target_col=target_col,
                    title=title,
                    class_names=tuple(class_names.items()),
                )
                st.pyplot(fig)

            elif problem_type == ProblemType.Regression:
                st.markdown("**Continuous Distribution**")
                title = (
                    f"Continuous Distribution for {target_name}"
                    if target_name
                    else "Continuous Distribution"
                )
                st.pyplot(_plot_continuous_distribution(df=df, target_col=target_col, title=title))

            return df
