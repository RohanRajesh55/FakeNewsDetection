#!/usr/bin/env python
# dashboard.py
import streamlit as st
import pandas as pd
import os

st.title("Fake News Detection Monitoring Dashboard")

st.header("Feedback Overview")
feedback_log = "feedback.log"
if os.path.exists(feedback_log):
    # If you decide to include an optional fourth column (weight), adjust the column names accordingly.
    df = pd.read_csv(feedback_log, sep="\t", header=None, names=["News_Text", "Prediction", "Feedback", "Optional_Weight"])
    st.write("Total feedback records:", len(df))
    st.write("Feedback breakdown:")
    st.write(df["Feedback"].value_counts())
    st.subheader("Recent Feedback Entries")
    st.dataframe(df.head(20))
else:
    st.write("No feedback logs found.")

st.header("Recent Application Logs")
app_log = "app.log"
if os.path.exists(app_log):
    with open(app_log, "r") as f:
        logs = f.readlines()
    st.text_area("Last 50 Log Lines", "".join(logs[-50:]), height=300)
else:
    st.write("No application logs found.")

st.header("Instructions")
st.markdown("""
- Use this dashboard to monitor real-time feedback and log updates.
- If you see a lot of corrective feedback on certain news items, consider reviewing them.
- This view can help you decide when to trigger model promotion.
""")