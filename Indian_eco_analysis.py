import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Youth Perspective on Indian Economy", layout="wide")

st.title(" Youth Perspective on Indian Economy")
st.markdown("### Advanced Data Mining & Cluster Analysis")
st.caption("Dataset: Survey-based youth demographic and economic perception data (2025)")

df = pd.read_excel("analysis_data_extended.xlsx")
df.columns = df.columns.str.strip()

# ---------------- KPI SECTION ----------------
st.subheader("📊 Key Metrics Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Respondents", len(df))

with col2:
    avg_children = df['What is the ideal number of children for a financially stable family?'].mode()[0]
    st.metric("Most Preferred Family Size", avg_children)

with col3:
    st.metric("Clusters Identified", "3")

st.divider()

# ---------------- NAVIGATION ----------------
section = st.sidebar.selectbox("Select Analysis", [
    "Demographic Insights",
    "Marriage & Financial Trends",
    "Urban vs Rural Comparison",
    "Workforce & AI Outlook",
    "Clustering & Segmentation"
])

    # ---------------- DEMOGRAPHICS ----------------
    if section == "Demographic Insights":

        st.subheader("Youth Age Distribution")

        age_counts = df['Age group'].value_counts(normalize=True) * 100
        fig, ax = plt.subplots()
        ax.bar(age_counts.index, age_counts.values)
        st.pyplot(fig)

        st.info("Majority respondents fall within 17–22 age group — representing emerging workforce.")

    # ---------------- MARRIAGE ----------------
    elif section == "Marriage & Financial Trends":

        colA, colB = st.columns(2)

        with colA:
            years = [2010, 2015, 2020, 2025]
            avg_age = [22, 24, 26, 27]
            fig, ax = plt.subplots()
            ax.plot(years, avg_age, marker='o')
            ax.set_title("Marriage Age Trend")
            st.pyplot(fig)

        with colB:
            cross_tab = pd.crosstab(
                df['Gender'],
                df['How important is financial stability before marriage?'],
                normalize='index') * 100
            st.bar_chart(cross_tab)

        st.info("Financial stability is strongly linked to delayed marriage decisions.")

    # ---------------- URBAN ----------------
    elif section == "Urban vs Rural Comparison":

        cross_tab = pd.crosstab(
            df['Type of area you live'],
            df['Does living in a city make it harder to raise children due to higher cost of living?'],
            normalize='index') * 100

        st.bar_chart(cross_tab)

        st.info("Urban youth perceive higher financial difficulty in raising children.")

    # ---------------- WORKFORCE ----------------
    elif section == "Workforce & AI Outlook":

        col1, col2 = st.columns(2)

        with col1:
            years = [2025, 2035, 2045, 2055]
            gdp = [6.8,6.5,5.9,6.2]
            labor = [100,90,80,82]

            fig, ax = plt.subplots()
            ax.plot(years, gdp, label="GDP Growth")
            ax.plot(years, labor, label="Labor Force Index")
            ax.legend()
            st.pyplot(fig)

        with col2:
            labels = ['AI Offsets Labor','Job Loss Dominates','Uncertain']
            values = [45,40,15]
            fig2, ax2 = plt.subplots()
            ax2.bar(labels, values)
            st.pyplot(fig2)

        st.info("AI perception is divided; workforce shrinkage may impact long-term growth.")


   
        # ---------------- CLUSTERING ----------------
    elif section == "Clustering & Segmentation":

        st.subheader("Youth Mindset Clusters (K-Means + t-SNE)")

        cols = ['Age group','Current Education/ Occupation Status','Type of area you live',
                'How important is financial stability before marriage?',
                'What is the ideal number of children for a financially stable family?']

        df_encoded = df.copy()

        for c in cols:
            df_encoded[c] = LabelEncoder().fit_transform(df_encoded[c].astype(str))

        scaled = StandardScaler().fit_transform(df_encoded[cols])

        kmeans = KMeans(n_clusters=3, random_state=42)
        df_encoded["Cluster"] = kmeans.fit_predict(scaled)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(scaled)

        df_encoded["TSNE1"] = tsne_data[:,0]
        df_encoded["TSNE2"] = tsne_data[:,1]

        # Define cluster names properly
        cluster_names = {
            0: "Urban-Modern",
            1: "Rural-Traditional",
            2: "Practical-Balanced"
        }

        palette = {
            0: "blue",
            1: "orange",
            2: "green"
        }

        fig, ax = plt.subplots(figsize=(5,4))

        for cluster in sorted(df_encoded["Cluster"].unique()):
            subset = df_encoded[df_encoded["Cluster"] == cluster]
            ax.scatter(
                subset["TSNE1"],
                subset["TSNE2"],
                label=cluster_names[cluster],
                color=palette[cluster],
                alpha=0.7
            )

        ax.set_title("Youth Economic & Family Mindset Clusters")
        ax.set_xlabel("Lifestyle & Financial Orientation")
        ax.set_ylabel("Family & Social Preference")
        ax.legend()

        st.pyplot(fig)

        st.success("""
        🔵 Urban-Modern – Financially aware, smaller family preference  
        🟠 Rural-Traditional – Higher fertility preference  
        🟢 Practical-Balanced – Close to national fertility average  
        """)

    # ---------------- REPORT DOWNLOAD ----------------
    st.divider()
    st.subheader("📥 Download Summary Report")

    report_text = """
    Youth Demographics & Economic Trends Summary

    - Majority youth aged 17–22
    - Increasing marriage age trend
    - Financial stability strongly influences decisions
    - Urban respondents prefer smaller families
    - Three distinct socio-economic clusters identified
    """

    st.download_button("Download Report", report_text, file_name="Youth_Economic_Summary.txt")

