import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    """Exploratory data analysis for the cleaned Zomato dataset."""

    # ======================================
    # Load Cleaned Dataset
    # ======================================
    file_path = r"zomato_clean.csv"  # or "Data/zomato_clean.csv"
    df = pd.read_csv(file_path)

    # ======================================
    # Q1–Q3: Locations Related
    # ======================================
    print("Q1 to Q3 – Locations Related")

    # Q1: Top 10 locations based on restaurant count
    print("\nQ1: Top 10 locations based on restaurant count")
    top_locations_count = (
        df["location"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_locations_count.columns = ["location", "restaurant_count"]
    print(top_locations_count)

    # Q2: Top 10 locations based on total votes
    print("\nQ2: Top 10 locations based on total votes")
    top_locations_votes = (
        df.groupby("location")["votes"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_locations_votes.columns = ["location", "total_votes"]
    print(top_locations_votes)

    # Q3: Top 10 locations based on average cost
    print("\nQ3: Top 10 locations based on average cost")
    top_locations_cost = (
        df.groupby("location")["approx_cost(for two people)"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_locations_cost.columns = ["location", "avg_cost_for_two"]
    print(top_locations_cost)

    # ======================================
    # Q4–Q7: Services – Online Delivery & Table Booking
    # ======================================
    print("\nQ4 to Q7 – Services: Online Delivery & Table Booking")

    # Q4: Is online delivery service available?
    print("\nQ4: Is online delivery service available?")
    print(df["online_order"].value_counts())
    print((df["online_order"].value_counts(normalize=True) * 100).round(2))

    # Q5: Is table booking service available?
    print("\nQ5: Is table booking service available?")
    print(df["book_table"].value_counts())
    print((df["book_table"].value_counts(normalize=True) * 100).round(2))

    # Q6: Rating difference – Online orders vs No online
    print("\nQ6: Rating difference – Online orders vs No online")
    online_rating_compare = (
        df.groupby("online_order")["rate"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    online_rating_compare["mean"] = online_rating_compare["mean"].round(2)
    online_rating_compare["median"] = online_rating_compare["median"].round(2)
    print(online_rating_compare)

    # Q7: Rating difference – Table booking vs No booking
    print("\nQ7: Rating difference – Table booking vs No booking")
    table_rating_compare = (
        df.groupby("book_table")["rate"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    table_rating_compare["mean"] = table_rating_compare["mean"].round(2)
    table_rating_compare["median"] = table_rating_compare["median"].round(2)
    print(table_rating_compare)

    # ======================================
    # Q8–Q9: Ratings – Distribution & Location-wise
    # ======================================
    print("\nQ8 to Q9 – Ratings: Distribution & Location-wise")

    # Q8: Rating distribution (overall)
    print("\nQ8: Rating distribution (overall)")
    print(df["rate"].describe())
    print("\nRating categories count:")
    print(df["rating_category"].value_counts().sort_index())

    print("\nHistogram – Rating Distribution")
    plt.figure()
    df["rate"].hist(bins=20)
    plt.xlabel("Rating")
    plt.ylabel("Number of Restaurants")
    plt.title("Overall Rating Distribution")
    plt.show()

    # Q9: Location-wise rating
    print("\nQ9: Location-wise rating (Top 10 by mean rating)")
    location_rating_summary = (
        df.groupby("location")["rate"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    print(location_rating_summary.head(10))

    # ======================================
    # Q10–Q13: Restaurant Types & Services
    # ======================================
    print("\nQ10 to Q13 – Restaurant Types & Services")

    # Q10: Most common restaurant type
    print("\nQ10: Most common restaurant types (Top 10)")
    rest_type_counts = df["rest_type"].value_counts().head(10)
    print(rest_type_counts)

    # Q11: Average rating based on restaurant type
    print("\nQ11: Average rating based on restaurant type (Top 10 by mean rating)")
    rest_type_rating = (
        df.groupby("rest_type")["rate"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    print(rest_type_rating.head(10))

    # Optional: Filter restaurant types with at least 50 restaurants
    rest_type_rating_filtered = (
        rest_type_rating[rest_type_rating["count"] >= 50]
        .sort_values(by="mean", ascending=False)
    )

    print("\nQ11 (Filtered): Avg rating by restaurant type (count >= 50)")
    print(rest_type_rating_filtered.head(10))

    # Q12 & Q13: Different service types + relation with rating
    print("\nQ12 & Q13: Service type combination vs Rating")

    def get_service_type(row: pd.Series) -> str:
        """Derive a service type label from online_order and book_table flags."""
        if row["online_order"] == "Yes" and row["book_table"] == "Yes":
            return "Online + Table Booking"
        if row["online_order"] == "Yes":
            return "Only Online"
        if row["book_table"] == "Yes":
            return "Only Table Booking"
        return "No Online / No Booking"

    df["service_type"] = df.apply(get_service_type, axis=1)

    service_rating = (
        df.groupby("service_type")["rate"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    service_rating["mean"] = service_rating["mean"].round(2)
    service_rating["median"] = service_rating["median"].round(2)
    print(service_rating)

    # ======================================
    # Q14–Q15: Cost Distribution & Cost vs Rating
    # ======================================
    print("\nQ14 to Q15 – Cost Distribution & Cost vs Rating")

    # Q14: Cost distribution
    print("\nQ14: Cost distribution (numeric summary)")
    print(df["approx_cost(for two people)"].describe())
    print("\nCost categories count:")
    print(df["cost_category"].value_counts())

    print("\nHistogram – Approx Cost for Two")
    plt.figure()
    df["approx_cost(for two people)"].hist(bins=30)
    plt.xlabel("Approx Cost for Two")
    plt.ylabel("Number of Restaurants")
    plt.title("Cost Distribution")
    plt.xlim(0, 2500)
    plt.show()

    # Q15: Cost vs Rating
    print("\nQ15: Cost vs Rating (by cost category)")
    cost_rating = (
        df.groupby("cost_category")["rate"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    cost_rating["mean"] = cost_rating["mean"].round(2)
    print(cost_rating)

    print("\nScatter – Approx Cost vs Rating")
    ax = df.plot.scatter(
        x="approx_cost(for two people)",
        y="rate",
        alpha=0.3,
    )
    ax.set_title("Cost vs Rating")
    ax.set_xlabel("Approx Cost for Two")
    ax.set_ylabel("Rating")
    plt.show()

    # ======================================
    # Q16–Q19: Popular Chains & Best Restaurants
    # ======================================
    print("\nQ16 to Q19 – Popular Chains & Best Restaurants")

    # Q16: Most popular casual dining restaurant chains
    print("\nQ16: Most popular Casual Dining chains (Top 10)")
    casual = df[df["rest_type"].str.contains("Casual Dining", na=False)]
    casual_chains = (
        casual.groupby("name")["location"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    casual_chains.columns = ["restaurant_name", "outlet_count"]
    print(casual_chains)

    # Q17: Most popular quick bites chains
    print("\nQ17: Most popular Quick Bites chains (Top 10)")
    quick_bites = df[df["rest_type"].str.contains("Quick Bites", na=False)]
    quick_bites_chains = (
        quick_bites.groupby("name")["location"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    quick_bites_chains.columns = ["restaurant_name", "outlet_count"]
    print(quick_bites_chains)

    # Q18: Most popular cafes
    print("\nQ18: Most popular Cafes (Top 10)")
    cafes = df[df["rest_type"].str.contains("Cafe", na=False)]
    cafe_chains = (
        cafes.groupby("name")["location"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    cafe_chains.columns = ["restaurant_name", "outlet_count"]
    print(cafe_chains)

    # Q19: Best restaurants – cheapest, highly rated, reliable
    print("\nQ19: Best restaurants – Cheapest but highly rated")
    cheapest_good = (
        df[df["rate"] >= 3.5]
        .sort_values(
            by=["approx_cost(for two people)", "rate"],
            ascending=[True, False],
        )[["name", "location", "rate", "votes", "approx_cost(for two people)"]]
        .head(10)
    )
    print(cheapest_good)

    print("\nQ19: Highly rated overall (Top 10)")
    top_rated = (
        df.sort_values(by=["rate", "votes"], ascending=[False, False])[
            ["name", "location", "rate", "votes", "approx_cost(for two people)"]
        ]
        .head(10)
    )
    print(top_rated)

    print("\nQ19: Reliable (votes >= 1000 and high rating)")
    reliable = df[df["votes"] >= 1000]
    reliable_best = (
        reliable.sort_values(by=["rate", "votes"], ascending=[False, False])[
            ["name", "location", "rate", "votes", "approx_cost(for two people)"]
        ]
        .head(10)
    )
    print(reliable_best)

    # ======================================
    # Q20–Q21: Cuisines & Top Chains
    # ======================================
    print("\nQ20 to Q21 – Cuisines & Top Chains")

    # Q20: Most popular cuisines
    print("\nQ20: Most popular cuisines (Top 15)")
    cuisine_exploded = (
        df.assign(cuisine=df["cuisines"].str.split(","))
        .explode("cuisine")
    )
    cuisine_exploded["cuisine"] = cuisine_exploded["cuisine"].str.strip()

    top_cuisines = cuisine_exploded["cuisine"].value_counts().head(15)
    print(top_cuisines)

    # Q21: Top restaurant chains overall
    print("\nQ21: Top restaurant chains overall (Top 15)")
    top_chains = (
        df.groupby("name")["location"]
        .count()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_chains.columns = ["restaurant_name", "outlet_count"]
    print(top_chains)


if __name__ == "__main__":
    main()
