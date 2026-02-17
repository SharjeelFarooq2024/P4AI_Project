from src.data_loader import load_and_combine_data
from src.data_integrity import (
    check_basic_info,
    check_duplicates,
    check_missing_values,
    check_infinite_values,
)
from src.eda import (
    analyze_target_distribution,
    analyze_feature_types,
    summarize_numeric,
    categorical_counts,
    plot_numeric_distributions,
    plot_boxplots,
    correlation_analysis,
)
from src.config import TARGET_COLUMN


def main():
    df = load_and_combine_data()

    # Step 1: Basic Info
    check_basic_info(df)

    # Step 2: Integrity Checks
    check_duplicates(df)
    check_missing_values(df)
    check_infinite_values(df)

    # Step 3: Target Analysis
    analyze_target_distribution(df, TARGET_COLUMN)

    # Step 4: Feature Type Analysis
    categorical_cols, numeric_cols = analyze_feature_types(df)

    # Step 5: Numeric and Categorical Summaries
    summarize_numeric(df, numeric_cols)
    categorical_counts(df, categorical_cols)

    # Step 6: Numeric Distributions and simple boxplots
    plot_numeric_distributions(df, numeric_cols)
    plot_boxplots(df, numeric_cols)

    # Step 7: Correlation
    correlation_analysis(df, numeric_cols)


if __name__ == "__main__":
    main()
