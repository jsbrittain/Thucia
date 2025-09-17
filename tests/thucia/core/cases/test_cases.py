import pandas as pd
from thucia.core.cases import aggregate_cases
from thucia.core.cases import cases_per_month


def test_cases_per_month_fills_zeros():
    # Sample data with missing province-date combos
    data = {
        "Province": ["A", "A", "B"],
        "Date": ["2025-01-15", "2025-02-20", "2025-01-10"],
        "Status": ["active", "active", "active"],
    }
    df = pd.DataFrame(data)

    result = cases_per_month(df, fill_column="Province")
    expected_dates = pd.to_datetime(["2025-01-31", "2025-02-28"])
    expected_index = pd.MultiIndex.from_product(
        [["A", "B"], expected_dates], names=["Province", "Date"]
    )
    result = result.set_index(["Province", "Date"]).sort_index()

    # Assert that Dates are periods, with month end frequency
    assert result.index.dtypes["Date"] == "period[M]"

    # Assert all expected combinations exist
    for combo in expected_index:
        assert combo in result.index, f"Missing combination {combo}"

    # Check case counts
    assert result.loc[("A", pd.Timestamp("2025-01-31")), "Cases"] == 1
    assert result.loc[("A", pd.Timestamp("2025-02-28")), "Cases"] == 1
    assert result.loc[("B", pd.Timestamp("2025-01-31")), "Cases"] == 1
    assert result.loc[("B", pd.Timestamp("2025-02-28")), "Cases"] == 0


def test_aggregate_cases_epiweek_monday():
    # Sample data with missing province-date combos
    data = {
        "Province": ["A", "A", "B"],
        "Date": ["2025-01-15", "2025-02-20", "2025-01-10"],
        "Status": ["active", "active", "active"],
    }
    df = pd.DataFrame(data)

    result = aggregate_cases(df, fill_column="Province", period="W-SAT")

    # Assert that Dates are periods, with month end frequency
    assert result["Date"].dtype == "period[W-SAT]"  # week end Saturday
    # Assert that weeks all start on Monday
    assert all(result["Date"].dt.start_time.dt.weekday == 6)  # 6 = Sunday


def test_aggregate_cases_epiweek_sunday():
    # Sample data with missing province-date combos
    data = {
        "Province": ["A", "A", "B"],
        "Date": ["2025-01-15", "2025-02-20", "2025-01-10"],
        "Status": ["active", "active", "active"],
    }
    df = pd.DataFrame(data)

    result = aggregate_cases(df, fill_column="Province", period="W-SUN")

    # Assert that Dates are periods, with month end frequency
    assert result["Date"].dtype == "period[W-SUN]"
    # Assert that weeks all start on Sunday
    assert all(result["Date"].dt.start_time.dt.weekday == 0)  # 0 = Monday
