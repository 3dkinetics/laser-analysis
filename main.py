import streamlit as st
import pandas as pd
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
        # Remove unnecessary conversion
        # df['time'] = pd.to_numeric(df['time'].str.replace(',', '.'), errors='coerce')

        # List of speed columns
        speed_cols = ["speed1", "speed2", "speed3"]

        default_speed_cols = ["speed2", "speed3"]
        # Allow user to select which speed columns to include
        selected_speeds = st.multiselect(
            "Select speed columns to include", speed_cols, default=default_speed_cols
        )

        # Process data
        df = process_data(df, mass, selected_speeds)

        # Display processed data
        st.subheader("Processed Data")
        st.write(df)

        # Plotting
        plot_data(df, selected_speeds)


def process_data(df, mass, selected_speeds):
    # Ensure time is sorted
    df = df.sort_values("time").reset_index(drop=True)

    # The columns should already be numeric due to decimal=','
    # If you want to ensure they are numeric, you can use:
    numeric_cols = ["time", "time-ref", "dist", "dist-ref"] + selected_speeds
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Loop over each selected speed column
    for speed_col in selected_speeds:
        # Calculate acceleration: difference in speed over difference in time
        df["acc_" + speed_col] = df[speed_col].diff() / df["time"].diff()

        # Calculate force: F = m * a
        df["force_" + speed_col] = mass * df["acc_" + speed_col]

        # Calculate force per mass: F/m (which is acceleration)
        df["force_per_mass_" + speed_col] = df["force_" + speed_col] / mass

        # Calculate work: W = F * displacement (using 'dist-ref' as displacement)
        df["work_" + speed_col] = df["force_" + speed_col] * df["dist-ref"]

        # Calculate power: P = F * v
        df["power_" + speed_col] = df["force_" + speed_col] * df[speed_col]

        # Calculate power per mass: P/m
        df["power_per_mass_" + speed_col] = df["power_" + speed_col] / mass

        # Calculate kinetic energy: KE = 0.5 * m * v^2
        df["ke_" + speed_col] = 0.5 * mass * df[speed_col] ** 2

    return df


def plot_data(df, selected_speeds):
    parameters = {
        "acc_": {"name": "Acceleration", "unit": "m/s²"},
        "force_": {"name": "Force", "unit": "N"},
        "force_per_mass_": {"name": "Force per Mass", "unit": "N/kg"},
        "work_": {"name": "Work", "unit": "J"},
        "power_": {"name": "Power", "unit": "W"},
        "power_per_mass_": {"name": "Power per Mass", "unit": "W/kg"},
        "ke_": {"name": "Kinetic Energy", "unit": "J"},
    }

    for param_prefix, param_info in parameters.items():
        param_name = param_info["name"]
        param_unit = param_info["unit"]
        st.subheader(f"{param_name} ({param_unit}) vs Time")
        # Prepare data
        data = pd.DataFrame({"time": df["time"]})
        for speed_col in selected_speeds:
            col_name = param_prefix + speed_col
            data[speed_col] = df[col_name]
        # Melt data
        data_melted = data.melt(
            id_vars=["time"],
            value_vars=selected_speeds,
            var_name="Speed",
            value_name=param_name,
        )
        # Plot
        fig = px.line(
            data_melted,
            x="time",
            y=param_name,
            color="Speed",
            labels={"time": "Time (s)", param_name: f"{param_name} ({param_unit})"},
            title=f"{param_name} vs Time",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
