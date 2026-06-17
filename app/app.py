import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# LOAD FILES
# =====================================

model = joblib.load("models/student_model.pkl")
encoders = joblib.load("models/encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

df = pd.read_csv("data/student_performance.csv")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🎓 Student Analytics Dashboard")

st.sidebar.markdown("### Project Information")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Dataset Size: 2000 Students")
st.sidebar.write("Features: 12")

# =====================================
# TITLE
# =====================================

st.title("🎓 Student Performance Prediction System")

st.markdown(
    """
    Predict student academic performance using attendance,
    study habits, assessment scores, participation level,
    stress level, and other educational indicators.
    """
)

st.divider()

# =====================================
# INPUT SECTION
# =====================================

st.header("📋 Student Information")

col1, col2, col3 = st.columns(3)

with col1:

    attendance = st.slider(
        "Attendance (%)",
        40,
        100,
        80
    )

    study_hours = st.slider(
        "Study Hours Per Day",
        1.0,
        10.0,
        5.0
    )

    assignment_score = st.slider(
        "Assignment Score",
        35,
        100,
        75
    )

    previous_test_score = st.slider(
        "Previous Test Score",
        30,
        100,
        70
    )

with col2:

    sleep_hours = st.slider(
        "Sleep Hours",
        4.0,
        9.0,
        7.0
    )

    screen_time = st.slider(
        "Screen Time (Hours)",
        1.0,
        10.0,
        4.0
    )

    projects_completed = st.slider(
        "Projects Completed",
        0,
        10,
        5
    )

with col3:

    participation = st.selectbox(
        "Participation Level",
        ["Low", "Medium", "High"]
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["No", "Yes"]
    )

    extra_classes = st.selectbox(
        "Extra Classes",
        ["No", "Yes"]
    )

    parent_education = st.selectbox(
        "Parent Education",
        ["School", "Graduate", "Postgraduate"]
    )

    stress_level = st.selectbox(
        "Stress Level",
        ["Low", "Medium", "High"]
    )

# =====================================
# PREDICTION
# =====================================

if st.button("🚀 Predict Performance", use_container_width=True):

    input_df = pd.DataFrame({
        "Attendance": [attendance],
        "Study_Hours": [study_hours],
        "Assignment_Score": [assignment_score],
        "Previous_Test_Score": [previous_test_score],
        "Sleep_Hours": [sleep_hours],
        "Participation_Level": [
            encoders["Participation_Level"].transform([participation])[0]
        ],
        "Internet_Access": [
            encoders["Internet_Access"].transform([internet_access])[0]
        ],
        "Extra_Classes": [
            encoders["Extra_Classes"].transform([extra_classes])[0]
        ],
        "Parent_Education": [
            encoders["Parent_Education"].transform([parent_education])[0]
        ],
        "Screen_Time": [screen_time],
        "Projects_Completed": [projects_completed],
        "Stress_Level": [
            encoders["Stress_Level"].transform([stress_level])[0]
        ]
    })

    prediction = model.predict(input_df)

    probabilities = model.predict_proba(input_df)

    result = target_encoder.inverse_transform(prediction)[0]

    confidence = probabilities.max() * 100

    st.success(f"🎯 Predicted Performance: {result}")

    st.info(
        f"📊 Prediction Confidence: {confidence:.2f}%"
    )

    st.subheader("Prediction Probabilities")

    classes = target_encoder.inverse_transform(
        range(len(probabilities[0]))
    )

    for cls, prob in zip(classes, probabilities[0]):

        st.write(f"**{cls}**")

        st.progress(float(prob))

        st.write(f"{prob * 100:.2f}%")

# =====================================
# ANALYTICS DASHBOARD
# =====================================

st.divider()

st.header("📊 Data Analytics Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Performance",
        "Study Hours",
        "Attendance",
        "Scores"
    ]
)

# =====================================
# PERFORMANCE TAB
# =====================================

with tab1:

    st.subheader("Performance Distribution")

    fig, ax = plt.subplots()

    df["Performance"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Performance")
    ax.set_ylabel("Students")

    st.pyplot(fig)

# =====================================
# STUDY HOURS TAB
# =====================================

with tab2:

    st.subheader("Study Hours Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df["Study_Hours"],
        bins=15
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Students")

    st.pyplot(fig)

# =====================================
# ATTENDANCE TAB
# =====================================

with tab3:

    st.subheader("Attendance Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df["Attendance"],
        bins=15
    )

    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Students")

    st.pyplot(fig)

# =====================================
# SCORES TAB
# =====================================

with tab4:

    col_a, col_b = st.columns(2)

    with col_a:

        st.subheader("Assignment Score Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            df["Assignment_Score"],
            bins=15
        )

        ax.set_xlabel("Assignment Score")
        ax.set_ylabel("Students")

        st.pyplot(fig)

    with col_b:

        st.subheader("Previous Test Score Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            df["Previous_Test_Score"],
            bins=15
        )

        ax.set_xlabel("Previous Test Score")
        ax.set_ylabel("Students")

        st.pyplot(fig)

# =====================================
# INSIGHTS
# =====================================

st.divider()

st.header("📈 Performance Insights")

st.info(
    """
    Top Influencing Factors:

    • Assignment Score

    • Previous Test Score

    • Attendance

    • Projects Completed

    • Study Hours
    """
)

st.success(
    """
    Suggestions to Improve Academic Performance:

    • Maintain attendance above 85%

    • Study consistently for 5–7 hours daily

    • Complete more practical projects

    • Reduce excessive screen time

    • Maintain a healthy sleep schedule
    """
)

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-Learn, Random Forest, Matplotlib and Streamlit"
)
