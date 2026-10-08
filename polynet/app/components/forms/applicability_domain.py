import streamlit as st

from polynet.app.options.state_keys import PredictPageStateKeys
from polynet.config.enums import ApplicabilityDomainMetric, MolecularDescriptor
from polynet.config.schemas import ApplicabilityDomainConfig
from polynet.config.schemas.fingerprints import MorganFingerprintConfig, SamplingFingerprintConfig


def applicability_domain_form() -> ApplicabilityDomainConfig:
    """Applicability domain settings for the predictions (defaults: assessed)."""
    defaults = ApplicabilityDomainConfig()

    with st.expander("Applicability domain", expanded=False):
        st.markdown(
            "Each new polymer is compared with the polymers the models were trained on, by "
            "its mean distance to its *k* nearest training polymers. It is **in domain** "
            "when that distance is at most ⟨d⟩ + Z·σ of the training polymers' own "
            "distances (Tropsha et al., 2003); the **score** is distance / cutoff, so 1 is "
            "the boundary. The **expected error** is the held-out test error of the models "
            "at a similar score (Sheridan et al., 2004); it is empty for polymers farther "
            "out than every test polymer."
        )
        enabled = st.checkbox(
            "Assess the applicability domain",
            value=defaults.enabled,
            key=PredictPageStateKeys.ADEnabled,
        )
        if not enabled:
            return ApplicabilityDomainConfig(enabled=False)

        metric = st.multiselect(
            "Distance",
            options=list(ApplicabilityDomainMetric),
            default=defaults.metric,
            key=PredictPageStateKeys.ADMetric,
            format_func=lambda m: {
                ApplicabilityDomainMetric.RuzickaMorgan: "Ruzicka · count fingerprint (all models)",
                ApplicabilityDomainMetric.EuclideanModelInputs: "Euclidean · model inputs (TML only)",
            }[m],
            help="Ruzicka: min–max similarity on the ratio-weighted count fingerprint, the same "
            "for every model. Euclidean: distance in each traditional model's scaled, "
            "feature-selected inputs, one domain per representation; GNNs always use Ruzicka. "
            "Each selected distance adds its own columns.",
        )
        if not metric:
            st.error("Select at least one distance, or untick the applicability domain.")
            st.stop()

        cols = st.columns(3)
        k_neighbours = cols[0].number_input(
            "Nearest neighbours (k)",
            min_value=1,
            value=defaults.k_neighbours,
            step=1,
            key=PredictPageStateKeys.ADNeighbours,
        )
        z = cols[1].number_input(
            "Cutoff Z",
            min_value=0.0,
            value=defaults.z,
            step=0.1,
            key=PredictPageStateKeys.ADZ,
            help="Larger Z widens the domain.",
        )
        n_bins = cols[2].number_input(
            "Similarity bins",
            min_value=1,
            value=defaults.n_bins,
            step=1,
            key=PredictPageStateKeys.ADBins,
            help="Bins of test-set domain score used to estimate the expected error.",
        )

        st.caption("Fingerprint of the Ruzicka distance")
        cols = st.columns(3)
        fingerprint = cols[0].selectbox(
            "Fingerprint",
            options=[MolecularDescriptor.Morgan, MolecularDescriptor.RDKitFP],
            key=PredictPageStateKeys.ADFingerprint,
            format_func=lambda f: f.value,
        )
        fp_size = cols[1].number_input(
            "Fingerprint size",
            min_value=1,
            value=defaults.fingerprint.fp_size,
            step=1,
            key=PredictPageStateKeys.ADFpSize,
        )
        radius = None
        if fingerprint == MolecularDescriptor.Morgan:
            radius = cols[2].number_input(
                "Radius",
                min_value=0,
                value=MorganFingerprintConfig().radius,
                step=1,
                key=PredictPageStateKeys.ADFpRadius,
            )

    return ApplicabilityDomainConfig(
        enabled=True,
        metric=metric,
        k_neighbours=k_neighbours,
        z=z,
        n_bins=n_bins,
        fingerprint=SamplingFingerprintConfig(
            fingerprint=fingerprint, fp_size=fp_size, radius=radius
        ),
    )
