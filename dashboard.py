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
segregation_df = pd.read_csv(os.path.join(cwd, "data", "segregation.csv"))
seg_violations_df = pd.read_csv(os.path.join(cwd, "data", "seg_adm_merged.csv"))
# infractions_df = pd.read_csv(os.path.join(cwd, "data", "infractions.csv"))

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


def setFigParams(fig, axis_title_font=20, axis_tick_font=16, bar_gap=0.4):
    fig.update_traces(marker=dict(line=dict(width=1)))
    fig.update_traces(marker_color="#707B7C")
    fig.update_yaxes(showgrid=True, gridcolor="#CCCCCC")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Segregation Instances",
        xaxis=dict(
            tickfont=dict(size=axis_tick_font, color="black")
        ),
        yaxis=dict(
            tickfont=dict(size=axis_tick_font, color="black")
        ),
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

    col1, col2 = st.columns([0.2, 1])

    with col1:
        # value_options = st.radio(
        #     "Segregation Instances:",
        #     ["Raw Count", "Percentage of Avg. Population"],
        #     index=1,
        # )
        time = ["By FY", "By Month and Year"]
        st.radio("Time", options=time, key="time_option")

    if st.session_state["time_option"] == time[0]:
        fy_counts = (
            segregation_df_filtered.groupby(["FY"])
            .size()
            .reset_index(name="Segregation Instances")
            .sort_values(by=["FY"], ascending=[True])
        )

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
    elif st.session_state["time_option"] == time[1]:
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

        with col2:
            fig = px.bar(
                ym_counts, x="Date", y="Segregation Instances", width=900, height=500
            )
            setFigParams(fig=fig, bar_gap=0.2)
            st.plotly_chart(fig, theme="streamlit", use_container_width=False)
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
