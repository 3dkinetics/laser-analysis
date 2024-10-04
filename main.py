import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page layout to wide
st.set_page_config(layout="wide")


def main():
    st.title("Athlete Performance Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Athlete mass input
        mass = st.number_input(
            "Enter the athlete mass (in kg):", min_value=0.0, value=70.0
        )

        # Read CSV
        df = pd.read_csv(uploaded_file, sep=";", decimal=",")

        # Allow user to input overall measurement distance for display purposes
        max_distance = st.number_input(
            "Enter the maximum distance for displaying results (in meters):",
            min_value=1.0,
            value=30.0,
        )

        # Toggle to show or trim data outside the range
        show_all_data = st.checkbox(
            "Show data outside the 0 to maximum distance range?", value=False
        )

        # List of speed columns
        speed_cols = ["speed1", "speed2", "speed3"]
        default_speed_cols = ["speed2", "speed3"]

        # Allow user to select which speed columns to include
        selected_speeds = st.multiselect(
            "Select speed columns to include", speed_cols, default=default_speed_cols
        )

        # Allow user to choose the x-axis for plots
        x_axis_option = st.radio(
            "Select the x-axis for the plots", options=["Time", "Distance (dist-ref)"]
        )

        # Process the full data
        df_processed = process_data(df, mass, selected_speeds)

        # Filter data only for display and plotting based on the max distance
        df_filtered_for_display = filter_for_display(
            df_processed, max_distance, show_all_data
        )

        # Display processed data (filtered)
        st.subheader("Processed Data (Filtered for Display)")
        st.write(df_filtered_for_display)

        # Generate table for metrics based on filtered data
        interval_df, avg_max_df = generate_interval_metrics(
            df_filtered_for_display, selected_speeds
        )

        # Display the tables
        st.subheader("Metrics at 5 Meter Intervals")
        st.write(interval_df)
        st.subheader("Average and Maximum Speeds")
        st.write(avg_max_df)

        # Plot the original graphs from plot_data (filtered)
        plot_data(df_filtered_for_display, selected_speeds, x_axis_option)

        # Plot positive and negative acceleration graphs on the same graph
        plot_acceleration_graphs(
            df_filtered_for_display, selected_speeds, x_axis_option
        )

        # Show bar chart of sum of positive and negative accelerations
        display_acceleration_sums(df_processed, selected_speeds)


def filter_for_display(df, max_distance, show_all_data):
    """
    Filters the DataFrame for display and plotting based on the max distance and the toggle to show all data.
    """
    if show_all_data:
        return df  # Return the original data for display without filtering

    # Filter data only for display between 0 and the specified maximum distance
    return df[(df["dist-ref"] >= 0) & (df["dist-ref"] <= max_distance)]


def process_data(df, mass, selected_speeds):
    # Ensure time is sorted
    df = df.sort_values("time").reset_index(drop=True)

    # The columns should already be numeric due to decimal=','
    numeric_cols = ["time", "time-ref", "dist", "dist-ref"] + selected_speeds
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Calculate distance between consecutive records (difference in 'dist-ref')
    df["distance_done"] = df["dist-ref"].diff()

    for speed_col in selected_speeds:
        # Calculate acceleration: difference in speed over difference in time
        df["acc_" + speed_col] = df[speed_col].diff() / df["time"].diff()

        # Calculate force: F = m * a
        df["force_" + speed_col] = mass * df["acc_" + speed_col]

        # Calculate force per mass: F/m (which is acceleration)
        df["force_per_mass_" + speed_col] = df["force_" + speed_col] / mass

        # Calculate work: W = F * displacement (using 'distance_done' as displacement)
        df["work_" + speed_col] = df["force_" + speed_col] * df["distance_done"]

        # Calculate power: P = F * v
        df["power_" + speed_col] = df["force_" + speed_col] * df[speed_col]

        # Calculate power per mass: P/m
        df["power_per_mass_" + speed_col] = df["power_" + speed_col] / mass

        # Calculate kinetic energy: KE = 0.5 * mass * v^2
        df["ke_" + speed_col] = 0.5 * mass * df[speed_col] ** 2

        # Ensure positive and negative acceleration columns are calculated
        df["positive_acc_" + speed_col] = df["acc_" + speed_col].apply(
            lambda x: x if x > 0 else 0
        )
        df["negative_acc_" + speed_col] = df["acc_" + speed_col].apply(
            lambda x: x if x < 0 else 0
        )

    return df


def generate_interval_metrics(df, selected_speeds):
    interval_distance = 5  # Interval of 5 meters
    max_distance = df["dist-ref"].max()
    distances = np.arange(0, max_distance + interval_distance, interval_distance)

    # Prepare metrics dataframe
    metrics = []
    for dist in distances:
        # Find the index of the row closest to the specified distance
        closest_idx = (df["dist-ref"] - dist).abs().idxmin()
        row = {
            "Distance (m)": dist,
            "Time (s)": df.loc[closest_idx, "time"],
        }
        for speed_col in selected_speeds:
            row["Velocity_" + speed_col] = df.loc[closest_idx, speed_col]
            row["Acceleration_" + speed_col] = df.loc[closest_idx, "acc_" + speed_col]
            row["Force_" + speed_col] = df.loc[closest_idx, "force_" + speed_col]

        metrics.append(row)

    interval_df = pd.DataFrame(metrics)

    # Calculating averages and max speeds
    avg_max_metrics = []
    for dist in distances[1:]:  # Skip 0 meters
        dist_range_df = df[df["dist-ref"] <= dist]
        row = {"Distance (m)": dist}
        for speed_col in selected_speeds:
            row["Avg Speed_" + speed_col] = dist_range_df[speed_col].mean()
            row["Max Speed_" + speed_col] = dist_range_df[speed_col].max()
        avg_max_metrics.append(row)

    avg_max_df = pd.DataFrame(avg_max_metrics)

    return interval_df, avg_max_df


def plot_data(df, selected_speeds, x_axis_option):
    parameters = {
        "acc_": {"name": "Acceleration", "unit": "m/s²"},
        "force_": {"name": "Force", "unit": "N"},
        "force_per_mass_": {"name": "Force per Mass", "unit": "N/kg"},
        "work_": {"name": "Work", "unit": "J"},
        "power_": {"name": "Power", "unit": "W"},
        "power_per_mass_": {"name": "Power per Mass", "unit": "W/kg"},
        "ke_": {"name": "Kinetic Energy", "unit": "J"},
    }

    # Set x-axis label and data column based on user selection
    if x_axis_option == "Time":
        x_axis_label = "Time (s)"
        x_axis_column = "time"
    else:
        x_axis_label = "Distance (m)"
        x_axis_column = "dist-ref"

    for param_prefix, param_info in parameters.items():
        param_name = param_info["name"]
        param_unit = param_info["unit"]
        st.subheader(f"{param_name} ({param_unit}) vs {x_axis_label}")

        # Prepare data
        data = pd.DataFrame({x_axis_column: df[x_axis_column]})
        for speed_col in selected_speeds:
            col_name = param_prefix + speed_col
            data[speed_col] = df[col_name]

        # Melt data
        data_melted = data.melt(
            id_vars=[x_axis_column],
            value_vars=selected_speeds,
            var_name="Speed",
            value_name=param_name,
        )

        # Plot
        fig = px.line(
            data_melted,
            x=x_axis_column,
            y=param_name,
            color="Speed",
            labels={
                x_axis_column: x_axis_label,
                param_name: f"{param_name} ({param_unit})",
            },
            title=f"{param_name} vs {x_axis_label}",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_acceleration_graphs(df, selected_speeds, x_axis_option):
    st.subheader("Acceleration (Positive and Negative) vs Time/Distance for All Speeds")

    # Ensure df is a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Determine x-axis
    x_axis_label = "Time (s)" if x_axis_option == "Time" else "Distance (m)"
    x_axis_column = "time" if x_axis_option == "Time" else "dist-ref"

    # Prepare data for plotting
    data = pd.DataFrame({x_axis_column: df[x_axis_column]})

    for speed_col in selected_speeds:
        # Calculate positive and negative accelerations
        df["positive_acc_" + speed_col] = df["acc_" + speed_col].apply(
            lambda x: x if x > 0 else 0
        )
        df["negative_acc_" + speed_col] = df["acc_" + speed_col].apply(
            lambda x: x if x < 0 else 0
        )

        # Add positive and negative acceleration to the data for plotting
        data["positive_acc_" + speed_col] = df["positive_acc_" + speed_col]
        data["negative_acc_" + speed_col] = df["negative_acc_" + speed_col]

    # Melt the data to get it in the right shape for Plotly
    data_melted = data.melt(
        id_vars=[x_axis_column],
        value_vars=[f"positive_acc_{speed}" for speed in selected_speeds]
        + [f"negative_acc_{speed}" for speed in selected_speeds],
        var_name="Type",
        value_name="Acceleration",
    )

    # Define the color for each type (positive or negative)
    color_map = {f"positive_acc_{speed}": "blue" for speed in selected_speeds}
    color_map.update({f"negative_acc_{speed}": "red" for speed in selected_speeds})

    # Plot the scatter plot (dots only)
    fig = px.scatter(
        data_melted,
        x=x_axis_column,
        y="Acceleration",
        color="Type",
        labels={x_axis_column: x_axis_label, "Acceleration": "Acceleration (m/s²)"},
        title=f"Acceleration (Positive and Negative) vs {x_axis_label}",
        color_discrete_map=color_map,
    )
    fig.update_traces(marker=dict(size=6), mode="markers")  # Dots only, no lines

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


def display_acceleration_sums(df, selected_speeds):
    st.subheader("Sum of Positive and Negative Accelerations")

    sum_data = []

    for speed_col in selected_speeds:
        # Check if the positive and negative acceleration columns exist
        pos_acc_col = "positive_acc_" + speed_col
        neg_acc_col = "negative_acc_" + speed_col

        if pos_acc_col in df.columns and neg_acc_col in df.columns:
            positive_sum = df[pos_acc_col].sum()
            negative_sum = df[neg_acc_col].sum()

            # Append the sums to the data for bar chart
            sum_data.append(
                {"Speed": speed_col, "Acceleration": "Positive", "Sum": positive_sum}
            )
            sum_data.append(
                {"Speed": speed_col, "Acceleration": "Negative", "Sum": negative_sum}
            )

    if sum_data:
        # Create a DataFrame for the sums
        sum_df = pd.DataFrame(sum_data)

        # Plot the bar chart
        fig = px.bar(
            sum_df,
            x="Speed",
            y="Sum",
            color="Acceleration",
            barmode="group",
            labels={"Sum": "Sum of Accelerations (m/s²)"},
            title="Sum of Positive and Negative Accelerations",
            color_discrete_map={"Positive": "blue", "Negative": "red"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No positive or negative acceleration data available.")


if __name__ == "__main__":
    main()
