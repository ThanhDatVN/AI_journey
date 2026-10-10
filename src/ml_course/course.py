"""Cấu trúc khoá học: phase → chapter. Đây là nguồn sự thật duy nhất cho thứ tự học."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    key: str
    title: str
    chapters: tuple[str, ...]


PHASES: tuple[Phase, ...] = (
    Phase("foundations", "Foundations", ("00_course_orientation", "01_python_for_ml")),
    Phase("mathematics", "Mathematics", ("02_mathematics_for_ml",)),
    Phase(
        "data",
        "Data & ML Fundamentals",
        ("03_data_analysis_and_eda", "04_data_preprocessing", "05_machine_learning_fundamentals"),
    ),
    Phase(
        "supervised",
        "Supervised Learning",
        (
            "06_linear_regression",
            "07_logistic_regression",
            "08_knn",
            "09_naive_bayes",
            "10_decision_trees",
            "11_random_forest",
            "12_gradient_boosting",
            "13_svm_and_kernels",
        ),
    ),
    Phase(
        "practice",
        "Evaluation & Practical ML",
        (
            "14_model_evaluation",
            "15_cross_validation",
            "16_hyperparameter_optimization",
            "17_feature_engineering",
            "18_feature_selection",
            "19_imbalanced_learning",
        ),
    ),
    Phase(
        "unsupervised",
        "Unsupervised Learning",
        ("20_clustering", "21_dimensionality_reduction", "22_anomaly_detection"),
    ),
    Phase(
        "advanced",
        "Advanced Classical ML",
        (
            "23_probabilistic_machine_learning",
            "24_ensemble_learning",
            "25_time_series",
            "26_recommender_systems",
        ),
    ),
    Phase(
        "deep",
        "Deep Learning",
        (
            "27_deep_learning_foundations",
            "28_neural_networks",
            "29_cnn",
            "30_sequence_models",
            "31_transformers",
            "32_nlp",
            "33_computer_vision",
            "34_representation_learning",
            "35_reinforcement_learning",
        ),
    ),
    Phase(
        "trustworthy",
        "Trustworthy ML",
        ("36_explainable_ai", "37_uncertainty_and_calibration", "38_causal_machine_learning"),
    ),
    Phase("systems", "ML Systems & MLOps", ("39_ml_systems", "40_mlops", "41_capstone_projects")),
)

CHAPTER_TITLES: dict[str, str] = {
    "00_course_orientation": "Course Orientation",
    "01_python_for_ml": "Python for ML",
    "02_mathematics_for_ml": "Mathematics for ML",
    "03_data_analysis_and_eda": "Data Analysis & EDA",
    "04_data_preprocessing": "Data Preprocessing",
    "05_machine_learning_fundamentals": "ML Fundamentals",
    "06_linear_regression": "Linear Regression",
    "07_logistic_regression": "Logistic Regression",
    "08_knn": "K-Nearest Neighbors",
    "09_naive_bayes": "Naive Bayes",
    "10_decision_trees": "Decision Trees",
    "11_random_forest": "Random Forest",
    "12_gradient_boosting": "Gradient Boosting",
    "13_svm_and_kernels": "SVM & Kernels",
    "14_model_evaluation": "Model Evaluation",
    "15_cross_validation": "Cross Validation",
    "16_hyperparameter_optimization": "Hyperparameter Optimization",
    "17_feature_engineering": "Feature Engineering",
    "18_feature_selection": "Feature Selection",
    "19_imbalanced_learning": "Imbalanced Learning",
    "20_clustering": "Clustering",
    "21_dimensionality_reduction": "Dimensionality Reduction",
    "22_anomaly_detection": "Anomaly Detection",
    "23_probabilistic_machine_learning": "Probabilistic ML",
    "24_ensemble_learning": "Ensemble Learning",
    "25_time_series": "Time Series",
    "26_recommender_systems": "Recommender Systems",
    "27_deep_learning_foundations": "Deep Learning Foundations",
    "28_neural_networks": "Neural Networks",
    "29_cnn": "Convolutional Neural Networks",
    "30_sequence_models": "Sequence Models",
    "31_transformers": "Transformers",
    "32_nlp": "Natural Language Processing",
    "33_computer_vision": "Computer Vision",
    "34_representation_learning": "Representation Learning",
    "35_reinforcement_learning": "Reinforcement Learning",
    "36_explainable_ai": "Explainable AI",
    "37_uncertainty_and_calibration": "Uncertainty & Calibration",
    "38_causal_machine_learning": "Causal ML",
    "39_ml_systems": "ML Systems",
    "40_mlops": "MLOps",
    "41_capstone_projects": "Capstone Projects",
}

CHAPTERS: tuple[str, ...] = tuple(ch for phase in PHASES for ch in phase.chapters)


def chapter_label(chapter: str) -> str:
    """'06_linear_regression' → '06 · Linear Regression'."""
    number = chapter.split("_", 1)[0]
    return f"{number} · {CHAPTER_TITLES.get(chapter, chapter)}"


def next_chapter(chapter: str) -> str | None:
    if chapter not in CHAPTERS:
        return None
    idx = CHAPTERS.index(chapter)
    return CHAPTERS[idx + 1] if idx + 1 < len(CHAPTERS) else None


def phase_of(chapter: str) -> Phase | None:
    for phase in PHASES:
        if chapter in phase.chapters:
            return phase
    return None
