"""UI components for the DR AI Assessment System."""

from .sidebar import render_sidebar
from .diagnosis import render_diagnosis_section
from .progression import render_progression_section
from .gradcam import render_gradcam_section
from .shap_analysis import render_shap_section
from .evaluation import render_evaluation_section

__all__ = [
    "render_sidebar",
    "render_diagnosis_section",
    "render_progression_section",
    "render_gradcam_section",
    "render_shap_section",
    "render_evaluation_section",
]