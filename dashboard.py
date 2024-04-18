import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import os
import plotly.express as px

# Load Page Icon
cwd = os.getcwd()
im = Image.open(os.path.join(cwd, "assets", "umn_icon.jpg"))

# Load Data
segregation_df = pd.read_csv(os.path.join(cwd, "data", "unique_segregations.csv"))
seg_violations_df = pd.read_csv(
    os.path.join(cwd, "data", "violations_leading_to_segregation.csv")
)
population_df = pd.read_csv(os.path.join(cwd, "data", "population.csv"))

seg_violations_df.rename(columns={"PenaltyDays": "SegDays"}, inplace=True)

st.set_page_config(
    page_title="Segregation Study Dashboard",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<h1 style='text-align: center; color: black;'>Overview of Segregation at Correctional Facilities in Minnesota</h1>",
    unsafe_allow_html=True,
)


def setFigParams(
    fig,
    title_yaxis="Segregation Instances",
    axis_title_font=20,
    axis_tick_font=16,
    bar_gap=0.4,
):
    fig.update_traces(marker=dict(line=dict(width=1)))
    fig.update_traces(marker_color="#707B7C")
    fig.update_yaxes(showgrid=True, gridcolor="#CCCCCC")
    fig.update_layout(
        xaxis_title="",
        yaxis_title=title_yaxis,
        xaxis=dict(tickfont=dict(size=axis_tick_font, color="black")),
        yaxis=dict(tickfont=dict(size=axis_tick_font, color="black")),
        xaxis_title_font=dict(size=axis_title_font),
        yaxis_title_font=dict(size=axis_title_font, color="black"),
        bargap=bar_gap,
    )


def filter_dataframe(df, filters):
    df_filterd = df.copy()
    for column, value in filters.items():
        if value != "All" and column in df_filterd.columns:
            df_filterd = df_filterd[df_filterd[column] == value]
    return df_filterd


def top_n_rules(df: pd.DataFrame, n: int):
    """
    Calculate the top n rules for each unique level in a categorical column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: A DataFrame containing the top n rules with the highest percentages
    """
    rule_counts = df.groupby(["RuleTitle"]).size().reset_index(name="Instances")

    rule_counts["% Total Instances"] = (
        rule_counts["Instances"] / rule_counts["Instances"].sum() * 100
    ).round(2)

    rule_counts = rule_counts.sort_values(
        by="% Total Instances", ascending=False, ignore_index=True
    )

    if n == None:
        top_n_rule_counts = rule_counts.copy()
    else:
        top_n_rule_counts = rule_counts.copy().head(n)

    return top_n_rule_counts


# Function to get fiscal year based on date
def get_fiscal_year(date):
    for fy_key, fy_value in fy.items():
        start_date = pd.to_datetime(fy_value[0])
        end_date = pd.to_datetime(fy_value[1])
        if start_date <= date < end_date:
            return fy_key
    return None


viz = ["Segregation Over Time", "Segregation Period", "Rule Violations"]

with st.sidebar:
    st.selectbox(
        options=viz,
        index=0,
        label="Select Theme",
        key="viz_option",
    )
    st.divider()

if st.session_state["viz_option"] == viz[0]:
    with st.sidebar:
        facilities = sorted(segregation_df["RHUnit"].unique().tolist())
        facilities.insert(0, "All")

        st.selectbox(
            options=facilities, index=0, label="Facility", key="facility_option"
        )

    filters = {
        "RHUnit": st.session_state["facility_option"],
    }
    segregation_df_filtered = filter_dataframe(segregation_df, filters)

    population_df_filtered = population_df[
        population_df["RHUnit"].isin(segregation_df_filtered["RHUnit"].unique())
    ]

    population_df_filtered = filter_dataframe(population_df_filtered, filters)

    col1, col2 = st.columns([0.2, 1])

    with col1:
        time = ["By FY", "By Month and Year"]
        st.radio("Time", options=time, key="time_option")

        seg_instances = ["Count", "Percentage of Total Population"]
        st.radio(
            "Segregation Instances",
            options=seg_instances,
            key="seg_instance_option",
        )

        fy_counts = (
            segregation_df_filtered.groupby(["FY"])
            .size()
            .reset_index(name="Segregation Instances")
            .sort_values(by=["FY"], ascending=[True])
        )

        fy_population = (
            population_df_filtered.groupby(["FY"])["Avg Population"].sum().reset_index()
        )

        segregation_df_filtered["SegStartDate"] = pd.to_datetime(
            segregation_df_filtered["SegStartDate"]
        )
        segregation_df_filtered["Date"] = segregation_df_filtered[
            "SegStartDate"
        ].dt.strftime("%Y-%m")

        ym_counts = (
            segregation_df_filtered.groupby(["Date"])
            .size()
            .reset_index(name="Segregation Instances")
            .sort_values(by=["Date"], ascending=[True])
        )
        fy = {
            "FY19": ["7/1/2018", "7/1/2019"],
            "FY20": ["7/1/2019", "7/1/2020"],
            "FY21": ["7/1/2020", "7/1/2021"],
            "FY22": ["7/1/2021", "7/1/2022"],
        }

        ym_counts["Date"] = pd.to_datetime(ym_counts["Date"])
        ym_counts["FY"] = ym_counts["Date"].apply(get_fiscal_year)

        fy_counts = pd.merge(fy_counts, fy_population, on=["FY"], how="inner")
        fy_counts["Segregation Rate"] = (
            fy_counts["Segregation Instances"] * 100 / fy_counts["Avg Population"]
        ).round(2)

        ym_counts = pd.merge(ym_counts, fy_population, on=["FY"], how="inner")
        ym_counts["Segregation Rate"] = (
            ym_counts["Segregation Instances"] * 100 / ym_counts["Avg Population"]
        ).round(2)

    if (
        st.session_state["time_option"] == time[0]
        and st.session_state["seg_instance_option"] == seg_instances[0]
    ):
        with col2:
            fig = px.bar(
                fy_counts,
                x="FY",
                y="Segregation Instances",
                width=700,
                height=500,
                text="Segregation Instances",
            )
            setFigParams(fig=fig)
            st.plotly_chart(fig, theme="streamlit", use_container_width=False)
            st.divider()
            st.write(
                "Every time an inmate is put in segregation, it is counted as an _instance of segregation_."
            )
    elif (
        st.session_state["time_option"] == time[0]
        and st.session_state["seg_instance_option"] == seg_instances[1]
    ):
        with col2:
            fig = px.bar(
                fy_counts,
                x="FY",
                y="Segregation Rate",
                width=700,
                height=500,
                text="Segregation Rate",
            )
            setFigParams(fig=fig, title_yaxis="Segregation Rate (%)")
            st.plotly_chart(fig, theme="streamlit", use_container_width=False)
            st.divider()
            st.write(
                "Every time an inmate is put in segregation, it is counted as an _instance of segregation_."
            )
    elif (
        st.session_state["time_option"] == time[1]
        and st.session_state["seg_instance_option"] == seg_instances[0]
    ):
        with col2:
            fig = px.bar(
                ym_counts, x="Date", y="Segregation Instances", width=900, height=500
            )
            setFigParams(fig=fig, bar_gap=0.2)
            st.plotly_chart(fig, theme="streamlit", use_container_width=False)
            st.write(
                "Every time an inmate is put in segregation, it is counted as an _instance of segregation_."
            )
    elif (
        st.session_state["time_option"] == time[1]
        and st.session_state["seg_instance_option"] == seg_instances[1]
    ):
        with col2:
            fig = px.bar(
                ym_counts,
                x="Date",
                y="Segregation Rate",
                width=900,
                height=500,
            )
            setFigParams(fig=fig, title_yaxis="Segregation Rate (%)")
            st.plotly_chart(fig, theme="streamlit", use_container_width=False)
            st.divider()
            st.write(
                "Every time an inmate is put in segregation, it is counted as an _instance of segregation_."
            )

elif st.session_state["viz_option"] == viz[1]:
    with st.sidebar:
        facilities = sorted(segregation_df["RHUnit"].unique().tolist())
        facilities.insert(0, "All")

        fy = segregation_df["FY"].unique().tolist()
        fy.insert(0, "All")

        rule_violations = sorted(seg_violations_df["RuleTitle"].unique().tolist())
        rule_violations.insert(0, "All")

        st.selectbox(
            options=facilities, index=0, label="Facility", key="facility_option"
        )
        st.selectbox(options=fy, index=0, label="FY", key="fy_option")
        st.selectbox(
            options=rule_violations,
            index=0,
            label="Rule Violations",
            key="rule_violations_option",
        )

        filters = {
            "RHUnit": st.session_state["facility_option"],
            "FY": st.session_state["fy_option"],
            "RuleTitle": st.session_state["rule_violations_option"],
        }

        if st.session_state["rule_violations_option"] != "All":
            seg_violations_df_filtered = filter_dataframe(seg_violations_df, filters)
        else:
            seg_violations_df_filtered = filter_dataframe(segregation_df, filters)

    _, col2, _ = st.columns([0.15, 1, 0.15])
    with col2:
        fig_box = px.box(
            seg_violations_df_filtered,
            y="SegDays",
            color="Race",
            width=1050,
            height=600,
        )
        fig_box.update_yaxes(showgrid=True, gridcolor="#CCCCCC")
        fig_box.update_layout(
            yaxis_title="Segregation Days",
            yaxis=dict(tickfont=dict(size=16, color="black")),
            yaxis_title_font=dict(size=20, color="black"),
        )
        st.plotly_chart(fig_box)

elif st.session_state["viz_option"] == viz[2]:
    with st.sidebar:
        facilities = sorted(seg_violations_df["RHUnit"].unique().tolist())
        facilities.insert(0, "All")

        fy = sorted(seg_violations_df["FY"].unique().tolist())
        fy.insert(0, "All")

        race = sorted(seg_violations_df["Race"].unique().tolist())
        race.insert(0, "All")

        age = sorted(seg_violations_df["AgeCat"].unique().tolist())
        age.insert(0, "All")

        st.selectbox(
            options=facilities, index=0, label="Facility", key="facility_option"
        )
        st.selectbox(options=fy, index=0, label="FY", key="fy_option")
        st.selectbox(options=race, index=0, label="Race", key="race_option")
        st.selectbox(
            options=age,
            index=0,
            label="Age Group",
            key="age_option",
        )

        filters = {
            "RHUnit": st.session_state["facility_option"],
            "FY": st.session_state["fy_option"],
            "Race": st.session_state["race_option"],
            "AgeCat": st.session_state["age_option"],
        }

        seg_violations_df_filtered = filter_dataframe(seg_violations_df, filters)

    col1, col2 = st.columns([0.7, 1])

    with col1:
        st.toggle(label="See Full Breakdown", value=True, key="breakdown_option")

    if not st.session_state["breakdown_option"]:
        with col2:
            top_5_rules_df = top_n_rules(seg_violations_df_filtered, n=5)
            labels = top_5_rules_df["RuleTitle"].to_list()
            values = top_5_rules_df["% Total Instances"].to_list()
            labels.insert(-1, "Other")
            values.insert(-1, 100 - sum(values))

            fig_pie = px.pie(
                names=labels,
                values=values,
                hole=0.5,
                width=1000,
                height=500,
                opacity=0.8,
            )
            st.plotly_chart(fig_pie)

    elif st.session_state["breakdown_option"]:
        with col2:
            top_5_rules_df = top_n_rules(seg_violations_df_filtered, n=5)
            labels = top_5_rules_df["RuleTitle"].to_list()
            values = top_5_rules_df["% Total Instances"].to_list()
            labels.insert(-1, "Other")
            values.insert(-1, 100 - sum(values))

            fig_pie = px.pie(
                names=labels,
                values=values,
                hole=0.5,
                width=1000,
                height=500,
                opacity=0.8,
            )
            st.plotly_chart(fig_pie)

        with col1:
            top_n_rules_df = top_n_rules(seg_violations_df_filtered, n=None)
            st.dataframe(data=top_n_rules_df, hide_index=True, use_container_width=True)

    st.divider()
    st.write(
        """
                The _Pie Chart_ represents the most frequently violated rules as a percentage of total rule violations.
                
                The category _Other_ encompasses less frequently committed violations. Refer to the table for a full breakdown.
                """
    )
