import pandas as pd
import numpy as np


def main() -> None:
    """Zomato dataset cleaning and preprocessing pipeline."""

    # ==============================
    # 0. Load Data
    # ==============================
    file_path = r"Data/zomato.xlsx"  # Ensure the file exists in the Data folder
    df = pd.read_excel(file_path)

    # ==============================
    # 1. Initial Overview
    # ==============================
    print("=== Initial Dataset Overview ===")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    print("\nDetailed info:")
    df.info()

    print("\nTop 15 columns with missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    # ==============================
    # 2. Keep Only Relevant Columns
    # ==============================
    core_columns = [
        "name",
        "address",
        "location",
        "rest_type",
        "cuisines",
        "approx_cost(for two people)",
        "rate",
        "votes",
        "online_order",
        "book_table",
        "listed_in(type)",
        "listed_in(city)",
        "dish_liked",
        "phone",
    ]

    final_columns = [col for col in core_columns if col in df.columns]
    df = df[final_columns]

    print("\n=== After Selecting Core Columns ===")
    print("Columns retained:")
    print(df.columns)
    print("New shape:", df.shape)

    print("\nInfo after column filtering:")
    df.info()

    print("\nTop 10 columns with missing values (after column filtering):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # ==============================
    # 3. Remove Duplicates
    # ==============================
    print("\nStep 1 – Remove duplicates")
    df.drop_duplicates(inplace=True)
    print("Shape after dropping duplicates:", df.shape)

    # ==============================
    # 4. Clean 'rate' Column
    # ==============================
    print("\nStep 2 – Clean 'rate' column")
    df["rate"] = df["rate"].str.replace("/5", "", regex=False).str.strip()
    df["rate"] = df["rate"].replace(["NEW", "-", ""], np.nan)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    # ==============================
    # 5. Clean 'approx_cost(for two people)'
    # ==============================
    print("\nStep 3 – Clean 'approx_cost(for two people)' column")
    df["approx_cost(for two people)"] = (
        df["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df["approx_cost(for two people)"] = pd.to_numeric(
        df["approx_cost(for two people)"], errors="coerce"
    )

    print(
        "Cost dtype after clean:",
        df["approx_cost(for two people)"].dtype,
    )

    # ==============================
    # 6. Clean 'votes'
    # ==============================
    print("\nStep 4 – Clean 'votes' column")
    df["votes"] = (
        df["votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df["votes"] = (
        pd.to_numeric(df["votes"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # ==============================
    # 7. Normalize Categorical Columns
    # ==============================
    print("\nStep 5 – Normalize categorical columns")
    df["online_order"] = df["online_order"].str.strip().str.title()
    df["book_table"] = df["book_table"].str.strip().str.title()
    df["location"] = df["location"].str.strip().str.title()
    df["rest_type"] = df["rest_type"].str.strip().str.title()
    df["cuisines"] = df["cuisines"].str.strip()

    # ==============================
    # 8. Handle Missing Values
    # ==============================
    print("\nStep 6 – Handle missing values (drop rows with missing rate or location)")
    df = df.dropna(subset=["rate", "location"])
    print("Shape after dropping missing critical values:", df.shape)

    # ==============================
    # 9. Derived Columns – Rating & Cost Categories
    # ==============================
    print("\nStep 7 – Create derived columns")

    # 9.1 Rating category
    df["rating_category"] = pd.cut(
        df["rate"],
        bins=[0, 2.5, 3.5, 4.5, 5],
        labels=["Poor", "Average", "Good", "Excellent"],
    )

    # 9.2 Cost category (initial binning)
    cost_max = df["approx_cost(for two people)"].max()

    df["cost_category"] = pd.cut(
        df["approx_cost(for two people)"],
        bins=[0, 500, 1500, 3000, cost_max],
        labels=["Low", "Medium", "High", "Premium"],
        include_lowest=True,
    )

    # 9.3 Fill missing cost_category values based on numeric cost
    def fill_cost_category(row):
        """Assign a cost category based on approx_cost if cost_category is missing."""
        if pd.isna(row["cost_category"]) and not pd.isna(
            row["approx_cost(for two people)"]
        ):
            cost = row["approx_cost(for two people)"]
            if cost <= 500:
                return "Low"
            if cost <= 1500:
                return "Medium"
            if cost <= 3000:
                return "High"
            return "Premium"
        return row["cost_category"]

    df["cost_category"] = df.apply(fill_cost_category, axis=1)

    # 9.4 If approx_cost itself is missing → mark cost_category as 'Unknown'
    df["cost_category"] = df.apply(
        lambda row: "Unknown"
        if pd.isna(row["approx_cost(for two people)"])
        else row["cost_category"],
        axis=1,
    )

    print("\nSample of cost vs cost_category:")
    print(df[["approx_cost(for two people)", "cost_category"]].head(10))

    print("\nCost category value counts:")
    print(df["cost_category"].value_counts())

    # ==============================
    # 10. Final Summary
    # ==============================
    print("\n=== Final Cleaned Dataset Summary ===")
    print("Final shape:", df.shape)

    print("\nTop 5 rows:")
    print(df.head(5))

    print("\nTop 5 columns with remaining missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(5))

    print("\nDescriptive statistics (numeric columns):")
    print(df.describe())

    # ==============================
    # 11. Save Clean Dataset
    # ==============================
    print("\nFinal Step – Save cleaned dataset")
    df.to_csv("zomato_clean.csv", index=False)
    print("Cleaned dataset saved as 'zomato_clean.csv'")


if __name__ == "__main__":
    main()
