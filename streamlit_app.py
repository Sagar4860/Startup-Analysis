import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- Page Configuration ---
st.set_page_config(page_title="🚀 Startup Funding Analysis", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f9fafc;
            padding: 2rem;
        }
        h1, h2, h3 {
            text-align: center;
            color: #333333;
            font-family: 'Segoe UI', sans-serif;
        }
        .stSelectbox > div, .stNumberInput > div {
            font-weight: 500;
        }
        .block-container {
            padding: 2rem 2rem;
        }
        .dataframe th {
            background-color: #f0f2f6;
        }
        .css-1cpxqw2 edgvbvh3 {  /* Removes Streamlit watermark */
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# --- Page Header ---
st.markdown("<h1>🚀 Startup Funding Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;font-size:18px;'>Explore Indian startup funding trends using interactive visualizations</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Load and Preprocess Data ---
df = pd.read_csv('startupfinal.csv')
df2 = pd.read_csv('df_explodes123.csv')

df2['date'] = pd.to_datetime(df2['date'], format='mixed', errors='coerce')
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter
df["yearmonth"] = (df['date'].dt.year * 100) + df['date'].dt.month



def load_investor_details(investor):
    # Add CSS for metric styling
    st.markdown("""
    <style>
    .metric-block {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        margin: 0 auto;
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Basic investor stats
    total = df['investors'].nunique()
    max_funding = df.groupby('investors')['amount'].max().sort_values(ascending=False).iloc[0]
    avg_funding = df.groupby('investors')['amount'].sum().mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('🧑‍💼 Total Investors', f"{total}")
    with col2:
        st.metric('💸 Max Funding by Investor', f"{max_funding} Cr")
    with col3:
        st.metric('📊 Avg Funding per Investor', f"{round(avg_funding)} Cr")

    st.markdown("---")
    st.markdown("### 🕵️‍♂️ Last 5 Investments by this Investor")
    last5_df = df2[df2['investors'].str.contains(investor, na=False)][
        ['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']
    ].head()
    st.dataframe(last5_df, use_container_width=True)

    filtered_df = df2[df2['investors'].str.contains(investor, na=False)]
    total_investment = filtered_df['amount'].sum()
    num_startups = filtered_df['startup'].nunique()

    col4, col5 = st.columns(2)
    with col4:
        st.metric("📈 Total Investment Amount", f"{total_investment} Cr")
    with col5:
        st.metric("🏢 Number of Startups Funded", num_startups)

    st.markdown("---")
    st.markdown("### 🧬 Similar Startups in the Same Sector")
    filtered = df2[df2['investors'] == investor]['vertical']

    if not filtered.empty:
        vertical_value = filtered.values[0]
        x = df2[df2['vertical'] == vertical_value][['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].sample(4)
        st.dataframe(x, use_container_width=True)
    else:
        st.warning(f"No entry for investor: **{investor}** found.")

    st.markdown("---")
    # First row of charts
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.markdown("### 💼 Top 5 Startups by Investment")
        bigseries = (
            filtered_df.groupby('startup')['amount']
            .sum()
            .sort_values(ascending=False)
            .head()
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(bigseries.index, bigseries.values, color='#00aaff')
        ax.set_xlabel('Startup')
        ax.set_ylabel('Investment Amount (Cr)')
        ax.set_title('Top Investments', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### 🧭 Sector-wise Distribution")
        vertical_series = filtered_df.groupby('vertical')['amount'].sum()
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.pie(
            vertical_series,
            labels=vertical_series.index,
            autopct="%0.01f%%",
            startangle=90,
            colors=sns.color_palette("pastel")
        )
        ax1.set_title('Sector Breakdown')
        ax1.axis('equal')
        fig1.tight_layout()
        st.pyplot(fig1)

    st.markdown("---")
    # Second row of pie charts
    with st.container():
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### 🔄 Investment Rounds Distribution")
            round_series = filtered_df.groupby('round')['amount'].sum()
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.pie(
                round_series,
                labels=round_series.index,
                autopct="%0.01f%%",
                startangle=140,
                colors=sns.color_palette("Set2")
            )
            ax2.set_title('Breakdown by Rounds')
            ax2.axis('equal')
            fig2.tight_layout()
            st.pyplot(fig2)

        with col4:
            st.markdown("### 🌆 City-wise Funding Distribution")
            city_series = filtered_df.groupby('city')['amount'].sum()
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.pie(
                city_series,
                labels=city_series.index,
                autopct="%0.01f%%",
                startangle=140,
                colors=sns.color_palette("husl")
            )
            ax3.set_title('City-wise Investment Breakdown')
            ax3.axis('equal')
            fig3.tight_layout()
            st.pyplot(fig3)

    st.markdown("---")
    # Year-over-Year Trend
    st.markdown("### 📈 Year-over-Year Investment Trend")
    df2['year'] = df2['date'].dt.year
    year_series = (
        df2[df2['investors'].str.contains(investor, na=False)]
        .groupby('year')['amount']
        .sum()
    )
    fig4, ax4 = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(
        x=year_series.index,
        y=year_series.values,
        marker='o',
        linewidth=2,
        color='#2E8B57',
        ax=ax4
    )
    ax4.set_title('Yearly Funding Trend', fontsize=14)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Total Investment (in Cr)')
    ax4.grid(visible=True, linestyle='--', alpha=0.6)
    fig4.tight_layout()
    st.pyplot(fig4)



@st.cache_data

def load_overall_analysis():
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    sns.set_theme(style="whitegrid")
    st.title("📊 Overall Startup Funding Analysis")

    # Core Metrics
    total = round(df['amount'].sum())
    max_funding = df.groupby('startup')['amount'].max().max()
    avg_funding = df.groupby('startup')['amount'].sum().mean()
    num_startups = df['startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Investment", f"{total} Cr")
    col2.metric("🚀 Max Investment", f"{max_funding} Cr")
    col3.metric("📈 Avg Investment", f"{round(avg_funding)} Cr")
    col4.metric("🏢 Funded Startups", num_startups)

    # Most Funded Startups Table
    st.markdown("### 💼 Top 5 Most Funded Startups")
    df_sorted = df.sort_values(by='amount', ascending=False)
    st.dataframe(df_sorted[['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].head(5), use_container_width=True)

    # Year-Month Funding Trend
    st.markdown("### 📅 Funding Trend Over Time")
    year_month = df['yearmonth'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=year_month.index, y=year_month.values, palette="Blues_d", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Funding Activity Over Time", fontsize=16)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("No. of Fundings")
    st.pyplot(fig)

    # Top Startups by Count
    st.markdown("### 🏆 Top 20 Startups by Number of Fundings")
    top_startups = df['startup'].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=top_startups.index, y=top_startups.values, palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Most Active Startups", fontsize=16)
    st.pyplot(fig)

    # Top Startups by Total Amount
    st.markdown("### 💸 Top 10 Startups by Total Funding")
    startup_funding = df.groupby('startup')['amount'].sum().nlargest(10).reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=startup_funding, x='startup', y='amount', palette="viridis", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Top Funded Startups", fontsize=16)
    st.pyplot(fig)

    # Industry-wise
    st.markdown("### 🏭 Industry-Wise Funding Distribution")
    industry = df['vertical'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=industry.index, y=industry.values, palette="Set2", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Top Funded Industry Verticals", fontsize=16)
    st.pyplot(fig)

    # City-wise
    st.markdown("### 🌆 City-wise Funding Analysis")
    city = df['city'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=city.index, y=city.values, palette="pastel", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Top Funded Cities", fontsize=16)
    st.pyplot(fig)

    # WordCloud of Investors
    st.markdown("### ☁️ Most Frequent Investors")
    names = df["investors"].dropna()
    wordcloud = WordCloud(max_font_size=60, width=800, height=400, colormap="Dark2").generate(' '.join(names))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Top Investors by Count
    st.markdown("### 🏦 Top 10 Investors by Number of Deals")
    investors = df['investors'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=investors.index, y=investors.values, palette="RdYlBu", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Most Active Investors", fontsize=16)
    st.pyplot(fig)

    # Top Investors by Total Funding
    st.markdown("### 💼 Top 10 Investors by Investment Amount")
    investor_funding = df.groupby('investors')['amount'].sum().nlargest(10).reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x='investors', y='amount', data=investor_funding, palette="cubehelix", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Biggest Investors by Capital", fontsize=16)
    st.pyplot(fig)

    # Round Types
    st.markdown("### 🔄 Types of Funding Rounds")
    investment = df['round'].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=investment.index, y=investment.values, palette="coolwarm", ax=ax)
    ax.set_title("Popular Investment Rounds", fontsize=16)
    st.pyplot(fig)
def load_startup_details(startups):
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")

    # 🔧 Custom CSS for better layout
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        h1, h2, h3 { color: #333333; }
        .metric { background: #f9f9f9; padding: 1rem; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

    st.title(f"🚀 Analysis of **{startups}**")

    # 🎯 Filtered data
    df_startup = df[df['startup'].str.lower() == startups.lower()]
    df_startup2 = df2[df2['startup'].str.lower().str.contains(startups.lower(), na=False)]

    # 📌 Metrics
    total_rounds = df_startup.shape[0]
    max_funding = df_startup['amount'].max()
    avg_funding = df_startup['amount'].mean()
    unique_investors = df_startup2['investors'].str.split(',').explode().str.strip().nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔁 Total Rounds", total_rounds)
    col2.metric("💸 Max Investment", f"{max_funding} Cr")
    col3.metric("📈 Avg Investment", f"{round(avg_funding)} Cr")
    col4.metric("👥 Unique Investors", unique_investors)

    # 📅 Recent Funding Rounds
    st.markdown("### 📅 Recent Funding Rounds")
    st.dataframe(
        df_startup[['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']]
        .sort_values(by='date', ascending=False),
        use_container_width=True
    )

    # 🧬 Similar Startups
    vertical = df_startup['vertical'].dropna().values[0] if not df_startup.empty else None
    if vertical:
        st.markdown(f"### 🧬 Similar Startups in **{vertical}**")
        similar_startups = df[df['vertical'] == vertical].sample(min(5, len(df[df['vertical'] == vertical])))
        st.dataframe(similar_startups[['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']],
                     use_container_width=True)
    else:
        st.warning("Could not find vertical information for the selected startup.")

    # 📈 YoY Investment Trend
    st.markdown("### 📈 Year-over-Year Investment Trend")
    df['yearmonth'] = pd.to_datetime(df['date']).dt.to_period('M')
    yoy_data = (
        df_startup.groupby('yearmonth')['amount']
        .sum()
        .reset_index()
    )
    yoy_data['yearmonth'] = yoy_data['yearmonth'].astype(str)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=yoy_data, x='yearmonth', y='amount', marker='o', linewidth=2, color='#2E8B57', ax=ax1)
    ax1.set_title(f"YoY Funding Trend for {startups}", fontsize=14)
    ax1.set_xlabel("Year-Month")
    ax1.set_ylabel("Total Investment (Cr)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)

    # 🎯 Additional Insights
    st.markdown("### 🧭 Investment Round & City-Wise Distribution")
    col5, col6 = st.columns(2)

    # Round-wise investment
    with col5:
        round_dist = df_startup['round'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.pie(round_dist, labels=round_dist.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
        ax2.set_title("Round Distribution")
        st.pyplot(fig2)

    # City-wise investment
    with col6:
        city_dist = df_startup['city'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.pie(city_dist, labels=city_dist.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax3.set_title("City Distribution")
        st.pyplot(fig3)


df['investors']=df['investors'].fillna('Undisclosed')
st.sidebar.title('Startup Funding Analysis')

option = st.sidebar.selectbox('Select One',['Overall Analysis','Startup','Investor','Question Answer'])


if option == 'Overall Analysis':
    load_overall_analysis()

elif option == 'Startup':
    selected_startup = st.sidebar.selectbox('Select StartUp',sorted(set(df2['startup'].dropna().str.split(',').apply(lambda x: [str(i) for i in x]).sum())))
    btn1 = st.sidebar.button('Find StartUp Details')
    if btn1:
        load_startup_details(selected_startup)
elif option=='Investor':
    selected_investor = st.sidebar.selectbox('Select Investor',sorted(set(df2['investors'].dropna().str.split(',').apply(lambda x: [str(i) for i in x]).sum())))
    btn2 = st.sidebar.button('Find Investor Details')
    if btn2:
        load_investor_details(selected_investor)
else:
    # Inject CSS for improved UI
    st.markdown("""
    <style>
    .question-card {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .question-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .stSelectbox > div {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    st.markdown('<div class="question-title">🧠 Choose a Visualization</div>', unsafe_allow_html=True)

    questions = [
        'selectone',
        "1. Top 5 Most Funded Startups",
        "2. Year-Month Distribution of Fundings",
        "3. Top 20 Startups by Number of Fundings",
        "4. Top 10 Startups by Total Funding Received",
        "5. Top 10 Industry Verticals by Fundings",
        "6. Top 10 Cities by Number of Fundings",
        "7. Wordcloud of Key Investors",
        "8. Top 10 Investors by Number of Fundings",
        "9. Funding Rounds by Number of Fundings"
    ]
    selected_question = st.selectbox('Select Analysis Type', questions)
    st.markdown('</div>', unsafe_allow_html=True)

    # Now show chart sections based on selected question
    def show_chart(fig=None):
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        if fig:
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    if selected_question == questions[1]:
        df_sorted = df.sort_values(by='amount', ascending=False)
        st.markdown("#### 📌 Top 5 Most Funded Startups")
        st.dataframe(df_sorted[['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].head(5), use_container_width=True)

    elif selected_question == questions[2]:
        st.markdown("#### 📅 Year-Month Distribution of Fundings")
        year_month = df['yearmonth'].value_counts()
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=year_month.index, y=year_month.values, palette="Blues_d", ax=ax)
        ax.set_xlabel('Year-Month')
        ax.set_ylabel('No. of Fundings')
        ax.set_title("Year-Month Distribution of Fundings")
        ax.tick_params(axis='x', rotation=45)
        show_chart(fig)

    elif selected_question == questions[3]:
        st.markdown("#### 🚀 Top 20 Startups by Number of Fundings")
        startupname = df['startup'].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=startupname.index, y=startupname.values, palette="coolwarm", ax=ax)
        ax.set_title("Top 20 Startups")
        ax.set_ylabel("No. of Fundings")
        ax.set_xlabel("Startup")
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(startupname.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)
        show_chart(fig)

    elif selected_question == questions[4]:
        st.markdown("#### 💰 Top 10 Startups by Total Funding Received")
        startup_funding = df.groupby(['startup'])['amount'].sum().sort_values(ascending=False).head(10).reset_index()
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x='startup', y='amount', data=startup_funding, palette="viridis", ax=ax)
        ax.set_title("Top 10 Startups by Total Funding")
        ax.set_ylabel("Total Funding (Cr)")
        ax.set_xlabel("Startup")
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(startup_funding['amount']):
            ax.text(i, v, f"{v:.2f}", ha='center', fontsize=9)
        show_chart(fig)

    elif selected_question == questions[5]:
        st.markdown("#### 🏭 Top 10 Industry Verticals by Fundings")
        industry = df['vertical'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=industry.index, y=industry.values, palette="Set2", ax=ax)
        ax.set_title("Top 10 Industry Verticals")
        ax.set_ylabel("Fundings")
        ax.set_xlabel("Vertical")
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(industry.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)
        show_chart(fig)

    elif selected_question == questions[6]:
        st.markdown("#### 🏙️ Top 10 Cities by Number of Fundings")
        city = df['city'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=city.index, y=city.values, palette="pastel", ax=ax)
        ax.set_title("Top 10 Cities")
        ax.set_ylabel("Fundings")
        ax.set_xlabel("City")
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(city.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)
        show_chart(fig)

    elif selected_question == questions[7]:
        st.markdown("#### 🧾 Wordcloud of Key Investors")
        names = df["investors"].dropna()
        wordcloud = WordCloud(max_font_size=50, width=800, height=400, colormap="Dark2").generate(' '.join(names))
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title("Wordcloud of Investors", fontsize=20)
        show_chart(fig)

    elif selected_question == questions[8]:
        st.markdown("#### 🧑‍💼 Top 10 Investors by Number of Fundings")
        investors = df['investors'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=investors.index, y=investors.values, palette="RdYlBu", ax=ax)
        ax.set_title("Top 10 Investors")
        ax.set_ylabel("Fundings")
        ax.set_xlabel("Investor")
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(investors.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)
        show_chart(fig)

    elif selected_question == questions[9]:
        st.markdown("#### 🧮 Funding Rounds by Number of Fundings")
        investment = df['round'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(x=investment.index, y=investment.values, palette="coolwarm", ax=ax)
        ax.set_title("Funding Rounds")
        ax.set_ylabel("Fundings")
        ax.set_xlabel("Type")
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(investment.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)
        show_chart(fig)
