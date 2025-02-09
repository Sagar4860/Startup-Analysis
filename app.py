import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud



st.set_page_config(layout="wide",page_title='startup analysis')
df=pd.read_csv('startupfinal.csv')
df2=pd.read_csv('df_explodes123.csv')
df2['date'] = pd.to_datetime(df2['date'], format='mixed', errors='coerce')
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter
df["yearmonth"] = (pd.to_datetime(df['date'],format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(df['date'],format='%d/%m/%Y').dt.month)
# # st.dataframe(df)
def load_investor_details(investor):
    total = df['investors'].nunique()
    # max amount infused in a startup
    max_funding = df.groupby('investors')['amount'].max().sort_values(ascending=False).head(1).values[0]
    # avg ticket size
    avg_funding = df.groupby('investors')['amount'].sum().mean()


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Total Investors', str(total))
    with col2:
        st.metric('Max Funding By investor', str(max_funding))

    with col3:
        st.metric('Avg fuding byb investor', str(round(avg_funding)) + ' Cr')

    last5_df=df2[df2['investors'].str.contains(investor)][['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].head()
    st.dataframe(last5_df)

    filtered_df = df2[df2['investors'].str.contains(investor, na=False)]

    # Display Metrics
    total_investment = filtered_df['amount'].sum()
    num_startups = filtered_df['startup'].nunique()
    st.metric("Total Investment Amount", f"{total_investment}")
    st.metric("Number of Startups Invested In", num_startups)



    # Filter the DataFrame to find the vertical of the given startup
    filtered = df2[df2['investors'] == investor]['vertical']

    # Check if the entry is present
    if not filtered.empty:
        # Extract the first vertical value
        vertical_value = filtered.values[0]
        # Filter the DataFrame where the vertical matches
        x = df2[df2['vertical'] == vertical_value][['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].sample(4)
        st.dataframe(x)
    else:
        # Handle the case where the entry is not present
        print(f"Entry for startup '{investor}' not present in the DataFrame.")

    # Layout with Columns
    col1, col2 = st.columns(2)

    # Column 1: Biggest Investments and Sector Distribution
    with col1:
        # Biggest Investments
        bigseries = (
            filtered_df.groupby('startup')['amount']
            .sum()
            .sort_values(ascending=False)
            .head()
        )
        st.subheader('Top 5 Startups by Investment')
        fig, ax = plt.subplots(figsize=(8, 6.85))  # Uniform size for all charts in this column
        ax.bar(bigseries.index, bigseries.values, color='skyblue')
        ax.set_xlabel('Startup')
        ax.set_ylabel('Investment Amount')
        ax.set_title('Top Investments')
        st.pyplot(fig)

        # Sector Distribution
        vertical_series = (
            filtered_df.groupby('vertical')['amount'].sum()
        )
        st.subheader('Sector Distribution')
        fig1, ax1 = plt.subplots(figsize=(8, 5))  # Uniform size
        ax1.pie(
            vertical_series,
            labels=vertical_series.index,
            autopct="%0.01f%%",
            startangle=90,
            colors=plt.cm.tab10.colors
        )
        ax1.set_title('Sector Breakdown')
        st.pyplot(fig1)

    # Column 2: Rounds and Cities
    with col2:
        # Investment Rounds
        round_series = (
            filtered_df.groupby('round')['amount'].sum()
        )
        st.subheader('Investment Rounds')
        fig2, ax2 = plt.subplots(figsize=(8, 5))  # Uniform size
        ax2.pie(
            round_series,
            labels=round_series.index,
            autopct="%0.01f%%",
            startangle=90,
            colors=plt.cm.Paired.colors
        )
        ax2.set_title('Round Breakdown')
        st.pyplot(fig2)

        # City Distribution
        city_series = (
            filtered_df.groupby('city')['amount'].sum()
        )
        st.subheader('City-wise Investment')
        fig3, ax3 = plt.subplots(figsize=(8, 6))  # Uniform size
        ax3.pie(
            city_series,
            labels=city_series.index,
            autopct="%0.01f%%",
            startangle=90,
            colors=plt.cm.Set3.colors
        )
        ax3.set_title('City Breakdown')
        st.pyplot(fig3)

    # Year-over-Year Investment
    df2['year'] = df2['date'].dt.year
    year_series = (
        df2[df2['investors'].str.contains(investor, na=False)]
        .groupby('year')['amount']
        .sum()
    )
    st.subheader('Year-over-Year Investment Trend')
    fig4, ax4 = plt.subplots(figsize=(10, 4))  # Different size for the trend graph
    ax4.plot(
        year_series.index,
        year_series.values,
        marker='o',
        linestyle='-',
        color='green',
        label='Investment Amount'
    )
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Total Investment')
    ax4.set_title('YoY Investment Trend')
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)

@st.cache_data
def load_overall_analysis():
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")

    st.title('Overall Analysis')

    # Total invested amount
    total = round(df['amount'].sum())
    max_funding = df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0]
    avg_funding = df.groupby('startup')['amount'].sum().mean()
    num_startups = df['startup'].nunique()

    # Most funded startups
    df_sorted = df.sort_values(by='amount', ascending=False)
    st.subheader('Top 5 Most Funded Startups')
    st.dataframe(df_sorted[['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].head(5))

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Total Investment', f"{total} Cr")
    with col2:
        st.metric('Max Investment', f"{max_funding} Cr")
    with col3:
        st.metric('Average Investment', f"{round(avg_funding)} Cr")
    with col4:
        st.metric('Funded Startups', num_startups)



    # Year-Month distribution
    st.header('YOY Graph')
    year_month = df['yearmonth'].value_counts()
    plt.figure(figsize=(15, 6))
    sns.barplot(x=year_month.index, y=year_month.values, palette="Blues_d")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('Year-Month of Transaction', fontsize=12)
    plt.ylabel('Number of Fundings Made', fontsize=12)
    plt.title("Year-Month Distribution of Fundings", fontsize=16)
    st.pyplot(plt)

    # Top 20 Startups by funding
    startupname = df['startup'].value_counts().head(20)
    plt.figure(figsize=(15, 6))
    sns.barplot(x=startupname.index, y=startupname.values, palette="coolwarm")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('Startup Name', fontsize=12)
    plt.ylabel('Number of Fundings Made', fontsize=12)
    plt.title("Top 20 Startups by Number of Fundings", fontsize=16)
    for i, v in enumerate(startupname.values):
        plt.text(i, v + 1, str(v), ha='center', fontsize=9)
    st.pyplot(plt)

    # Total funding received by startups
    startup_funding = df.groupby(['startup'])['amount'].sum().sort_values(ascending=False).head(10).reset_index()
    plt.figure(figsize=(15, 6))
    sns.barplot(x='startup', y='amount', data=startup_funding, palette="viridis")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('Startup Name', fontsize=12)
    plt.ylabel('Total Funding Amount (Cr)', fontsize=12)
    plt.title("Top 10 Startups by Total Funding Received", fontsize=16)
    for i, v in enumerate(startup_funding['amount']):
        plt.text(i, v, f"{v:.2f}", ha='center', fontsize=9)
    st.pyplot(plt)

    # Industry verticals
    st.text('Which industries are favored by investors for funding?')
    industry = df['vertical'].value_counts().head(10)
    plt.figure(figsize=(15, 6))
    sns.barplot(x=industry.index, y=industry.values, palette="Set2")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('Industry Vertical', fontsize=12)
    plt.ylabel('Number of Fundings Made', fontsize=12)
    plt.title("Top 10 Industry Verticals by Fundings", fontsize=16)
    for i, v in enumerate(industry.values):
        plt.text(i, v + 1, str(v), ha='center', fontsize=9)
    st.pyplot(plt)

    # City-wise funding
    st.header('City-wise Analysis of Funding')
    city = df['city'].value_counts().head(10)
    plt.figure(figsize=(15, 6))
    sns.barplot(x=city.index, y=city.values, palette="pastel")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Number of Fundings Made', fontsize=12)
    plt.title("Top 10 Cities by Number of Fundings", fontsize=16)
    for i, v in enumerate(city.values):
        plt.text(i, v + 1, str(v), ha='center', fontsize=9)
    st.pyplot(plt)

    # Wordcloud for investors
    st.subheader('Key Investors in the Indian Ecosystem')
    names = df["investors"].dropna()
    wordcloud = WordCloud(max_font_size=50, width=800, height=400, colormap="Dark2").generate(' '.join(names))
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Wordcloud of Investors", fontsize=20)
    st.pyplot(plt)

    # Top investors by number of fundings
    investors = df['investors'].value_counts().head(10)
    plt.figure(figsize=(15, 6))
    sns.barplot(x=investors.index, y=investors.values, palette="RdYlBu")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('Investor Name', fontsize=12)
    plt.ylabel('Number of Fundings Made', fontsize=12)
    plt.title("Top 10 Investors by Number of Fundings", fontsize=16)
    for i, v in enumerate(investors.values):
        plt.text(i, v + 1, str(v), ha='center', fontsize=9)
    st.pyplot(plt)

    # Types of funding rounds
    st.subheader('Types of Funding Rounds')
    investment = df['round'].value_counts().head(8)
    plt.figure(figsize=(15, 6))
    sns.barplot(x=investment.index, y=investment.values, palette="coolwarm")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('Investment Type', fontsize=12)
    plt.ylabel('Number of Fundings Made', fontsize=12)
    plt.title("Funding Rounds by Number of Fundings", fontsize=16)
    for i, v in enumerate(investment.values):
        plt.text(i, v + 1, str(v), ha='center', fontsize=9)
    st.pyplot(plt)

    # Total funding by investors
    investor_funding = df.groupby(['investors'])['amount'].sum().sort_values(ascending=False).head(10).reset_index()
    plt.figure(figsize=(15, 6))
    sns.barplot(x='investors', y='amount', data=investor_funding, palette="cubehelix")
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel('Investor Name', fontsize=12)
    plt.ylabel('Total Funding Amount (Cr)', fontsize=12)
    plt.title("Top 10 Investors by Total Funding Amount", fontsize=16)
    for i, v in enumerate(investor_funding['amount']):
        plt.text(i, v, f"{v:.2f}", ha='center', fontsize=9)
    st.pyplot(plt)

    # st.title('Overall Analysis')
    #
    # # total invested amount
    # total = round(df['amount'].sum())
    # # max amount infused in a startup
    # max_funding = df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0]
    # # avg ticket size
    # avg_funding = df.groupby('startup')['amount'].sum().mean()
    # # total funded startups
    # num_startups = df['startup'].nunique()
    #
    # df_sorted = df.sort_values(by='amount', ascending=False)
    # st.subheader('most funded')
    # st.dataframe(df_sorted.head(5))
    #
    #
    #
    # col1,col2,col3,col4 = st.columns(4)
    #
    # with col1:
    #     st.metric('Total',str(total) + ' Cr')
    # with col2:
    #     st.metric('Max', str(max_funding) + ' Cr')
    #
    # with col3:
    #     st.metric('Avg',str(round(avg_funding)) + ' Cr')
    #
    # with col4:
    #     st.metric('Funded Startups',num_startups)
    #
    # st.header('YOY graph')
    # temp = df['yearmonth'].value_counts().sort_values(ascending=False).head(10)
    # year_month = df['yearmonth'].value_counts()
    #
    # # Plotting
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x=year_month.index, y=year_month.values, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('Year-Month of transaction', fontsize=12)
    # plt.ylabel('Number of fundings made', fontsize=12)
    # plt.title("Year-Month Distribution", fontsize=16)
    #
    # st.pyplot(plt)
    #
    #
    #
    #
    # startupname = df['startup'].value_counts().head(20)
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x=startupname.index, y=startupname.values, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('Startup Name', fontsize=12)
    # plt.ylabel('Number of fundings made', fontsize=12)
    # plt.title("Number of funding a startup got", fontsize=16)
    # st.pyplot(plt)
    #
    #
    # startupname = df.groupby(['startup'])['amount'].sum().sort_values(ascending=False).head(10).reset_index()
    #
    # # Plotting
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x='startup', y='amount', data=startupname, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('Startup Name', fontsize=12)
    # plt.ylabel('Total Funding Amount', fontsize=12)
    # plt.title("Total Funding Received by Startups", fontsize=16)
    #
    # # Display plot in Streamlit
    # st.pyplot(plt)
    #
    #
    #
    #
    # st.text('Which industries are favored by investors for funding ? (OR) Which type of companies got more easily funding ?')
    # industry = df['vertical'].value_counts().head(10)
    #
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x=industry.index, y=industry.values, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('Industry vertical of startups', fontsize=12)
    # plt.ylabel('Number of fundings made', fontsize=12)
    # plt.title("Industry vertical of startups with number of funding", fontsize=16)
    # st.pyplot(plt)
    # city = df['city'].value_counts().head(10)
    # st.header('Do cities play a major role in funding ? (OR) Which city has maximum startups ?')
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x=city.index, y=city.values, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('city location of startups', fontsize=12)
    # plt.ylabel('Number of fundings made', fontsize=12)
    # plt.title("city location of startups with number of funding", fontsize=16)
    # st.pyplot(plt)
    #
    #
    # st.subheader('Who is the important investors in the Indian Ecosystem?')
    # names = df["investors"][~pd.isnull(df["investors"])]
    # # print(names)
    # wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
    # plt.figure(figsize=(15, 8))
    # plt.imshow(wordcloud)
    # plt.title("Wordcloud for Investor Names", fontsize=35)
    # plt.axis("off")
    # st.pyplot(plt)
    #
    # investors = df['investors'].value_counts().head(10)
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x=investors.index, y=investors.values, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('Investors Names', fontsize=12)
    # plt.ylabel('Number of fundings made', fontsize=12)
    # plt.title("Investors Names with number of funding", fontsize=16)
    #
    #
    # st.pyplot(plt)
    # st.subheader('What are different types of funding for startups ?')
    # investment = df['round'].value_counts().head(8)
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x=investment.index, y=investment.values, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('Investment Type', fontsize=12)
    # plt.ylabel('Number of fundings made', fontsize=12)
    # plt.title("Investment Type with number of funding", fontsize=16)
    # st.pyplot(plt)
    #
    # startupname = df.groupby(['investors'])['amount'].sum().sort_values(ascending=False).head(10).reset_index()
    #
    # # Plotting
    # plt.figure(figsize=(15, 8))
    # sns.barplot(x='investors', y='amount', data=startupname, alpha=0.9)
    # plt.xticks(rotation='vertical')
    # plt.xlabel('Startup Name', fontsize=12)
    # plt.ylabel('Total Funding Amount', fontsize=12)
    # plt.title("Total Funding Received by investors", fontsize=16)
    #
    # # Display plot in Streamlit
    # st.pyplot(plt)


def load_startup_details(startups):
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set Seaborn theme for consistency
    sns.set_theme(style="whitegrid")

    st.title('Startup Analysis')

    # Sorting by funding amount


    # Metrics for the selected startup
    # Replace with user input if applicable
    total = df[df['startup'] == startups].shape[0]
    max_funding = df[df['startup'] == startups]['amount'].max()
    avg_funding = df[df['startup'] == startups]['amount'].mean()
    totalinvest = df2[df2['startup'].str.contains(startups, case=False)]['investors'].nunique()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Total Investments', str(total))
    with col2:
        st.metric('Max Investment', f"{max_funding} Cr")
    with col3:
        st.metric('Average Investment', f"{round(avg_funding)} Cr")
    with col4:
        st.metric('Total Unique Investors', str(totalinvest))

    # Display funding details for the startup
    df3 = df[df['startup'] == startups][['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].head()
    st.dataframe(df3, width=1300)

    # Vertical-related startups
    filtered = df[df['startup'] == startups]['vertical']

    if not filtered.empty:
        vertical_value = filtered.values[0]
        similar_startups = df[df['vertical'] == vertical_value][
            ['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].sample(4)
        st.subheader(f'Similar Startups to {startups}')
        st.dataframe(similar_startups)
    else:
        st.warning(f"No data found for startup '{startups}'.")

    # Year-over-Year Investment Trend
    df2['year'] = df2['date'].dt.year
    year_series = (
        df[df['startup'] == startups]
        .groupby('yearmonth')['amount']
        .sum()
    )

    st.subheader('Year-over-Year Investment Trend')

    # Plotting YoY Trend
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(
        year_series.index,
        year_series.values,
        marker='o',
        linestyle='-',
        color='green',
        label='Investment Amount'
    )
    ax4.set_xlabel('Year-Month', fontsize=12)
    ax4.set_ylabel('Total Investment (Cr)', fontsize=12)
    ax4.set_title(f'YoY Investment Trend for {startups}', fontsize=16)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='upper left', fontsize=10)

    # Adding value annotations
    for i, (x, y) in enumerate(zip(year_series.index, year_series.values)):
        ax4.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)

    st.pyplot(fig4)

    # Enhanced DataFrame and Graph Visuals
    st.subheader(f"Analysis of {startups}'s Investment Data")

    # st.title('Startup Analysis')
    # df_sorted = df.sort_values(by='amount', ascending=False)
    # st.subheader('most funded')
    # st.dataframe(df_sorted.head(5))
    # if st.checkbox('Show dataframe'):
    #     chart_data = df_sorted
    # # total invested amount
    # total = df[df['startup']==startups].shape[0]
    # print(total)
    # # max amount infused in a startup
    # max_funding = df[df['startup']==startups]['amount'].max()
    # # avg ticket size
    # avg_funding = df[df['startup']==startups]['amount'].mean()
    # # total funded startups
    # totalinvest=df2[df2['startup'].str.contains('ola')]['investors'].nunique()
    # col1, col2, col3, col4 = st.columns(4)
    #
    # with col1:
    #     st.metric('Total', str(total))
    # with col2:
    #     st.metric('Max', str(max_funding) + ' Cr')
    # with col3:
    #     st.metric('Avg', str(round(avg_funding)) + ' Cr')
    # with col4:
    #     st.metric('Total investors',str(totalinvest))
    # df3=df[df['startup']==startups][['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].head()
    # st.dataframe(df3,width=1300)
    #
    #
    # # Filter the DataFrame to find the vertical of the given startup
    # filtered = df[df['startup'] == startups]['vertical']
    #
    # # Check if the entry is present
    # if not filtered.empty:
    #     # Extract the first vertical value
    #     vertical_value = filtered.values[0]
    #     # Filter the DataFrame where the vertical matches
    #     x = df[df['vertical'] == vertical_value][['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].sample(4)
    #     st.subheader('Same startup as {}'.format(startups))
    #     st.dataframe(x)
    # else:
    #     # Handle the case where the entry is not present
    #     print(f"Entry for startup '{startups}' not present in the DataFrame.")
    #
    # df2['year'] = df2['date'].dt.year
    # year_series = (
    #     df[df['startup']==startups]
    #     .groupby('yearmonth')['amount']
    #     .sum()
    # )
    # st.subheader('Year-over-Year Investment Trend')
    # fig4, ax4 = plt.subplots(figsize=(10, 4))  # Different size for the trend graph
    # ax4.plot(
    #     year_series.index,
    #     year_series.values,
    #     marker='o',
    #     linestyle='-',
    #     color='green',
    #     label='Investment Amount'
    # )
    # ax4.set_xlabel('Year')
    # ax4.set_ylabel('Total Investment')
    # ax4.set_title('YoY Investment Trend')
    # ax4.grid(True)
    # ax4.legend()
    # st.pyplot(fig4)



df['investors']=df['investors'].fillna('Undisclosed')
st.sidebar.title('Startup Funding Analysis')
st.title('Which Startup You Want To search for?')
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
    # Selectbox for visualizations
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

    # Logic for each visualization
    if selected_question == questions[1]:  # Top 5 Most Funded Startups
        df_sorted = df.sort_values(by='amount', ascending=False)
        st.subheader('Top 5 Most Funded Startups')
        st.dataframe(df_sorted[['date', 'startup', 'vertical', 'city', 'investors', 'round', 'amount']].head().head(5))

    elif selected_question == questions[2]:  # Year-Month Distribution of Fundings
        year_month = df['yearmonth'].value_counts()
        plt.figure(figsize=(15, 6))
        sns.barplot(x=year_month.index, y=year_month.values, palette="Blues_d")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('Year-Month of Transaction', fontsize=12)
        plt.ylabel('Number of Fundings Made', fontsize=12)
        plt.title("Year-Month Distribution of Fundings", fontsize=16)
        st.pyplot(plt)

    elif selected_question == questions[3]:  # Top 20 Startups by Number of Fundings
        startupname = df['startup'].value_counts().head(20)
        plt.figure(figsize=(15, 6))
        sns.barplot(x=startupname.index, y=startupname.values, palette="coolwarm")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('Startup Name', fontsize=12)
        plt.ylabel('Number of Fundings Made', fontsize=12)
        plt.title("Top 20 Startups by Number of Fundings", fontsize=16)
        for i, v in enumerate(startupname.values):
            plt.text(i, v + 1, str(v), ha='center', fontsize=9)
        st.pyplot(plt)

    elif selected_question == questions[4]:  # Top 10 Startups by Total Funding Received
        startup_funding = df.groupby(['startup'])['amount'].sum().sort_values(ascending=False).head(10).reset_index()
        plt.figure(figsize=(15, 6))
        sns.barplot(x='startup', y='amount', data=startup_funding, palette="viridis")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('Startup Name', fontsize=12)
        plt.ylabel('Total Funding Amount (Cr)', fontsize=12)
        plt.title("Top 10 Startups by Total Funding Received", fontsize=16)
        for i, v in enumerate(startup_funding['amount']):
            plt.text(i, v, f"{v:.2f}", ha='center', fontsize=9)
        st.pyplot(plt)

    elif selected_question == questions[5]:  # Top 10 Industry Verticals by Fundings
        industry = df['vertical'].value_counts().head(10)
        plt.figure(figsize=(15, 6))
        sns.barplot(x=industry.index, y=industry.values, palette="Set2")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('Industry Vertical', fontsize=12)
        plt.ylabel('Number of Fundings Made', fontsize=12)
        plt.title("Top 10 Industry Verticals by Fundings", fontsize=16)
        for i, v in enumerate(industry.values):
            plt.text(i, v + 1, str(v), ha='center', fontsize=9)
        st.pyplot(plt)

    elif selected_question == questions[6]:  # Top 10 Cities by Number of Fundings
        city = df['city'].value_counts().head(10)
        plt.figure(figsize=(15, 6))
        sns.barplot(x=city.index, y=city.values, palette="pastel")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('City', fontsize=12)
        plt.ylabel('Number of Fundings Made', fontsize=12)
        plt.title("Top 10 Cities by Number of Fundings", fontsize=16)
        for i, v in enumerate(city.values):
            plt.text(i, v + 1, str(v), ha='center', fontsize=9)
        st.pyplot(plt)

    elif selected_question == questions[7]:  # Wordcloud of Key Investors
        names = df["investors"].dropna()
        wordcloud = WordCloud(max_font_size=50, width=800, height=400, colormap="Dark2").generate(' '.join(names))
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Wordcloud of Investors", fontsize=20)
        st.pyplot(plt)

    elif selected_question == questions[8]:  # Top 10 Investors by Number of Fundings
        investors = df['investors'].value_counts().head(10)
        plt.figure(figsize=(15, 6))
        sns.barplot(x=investors.index, y=investors.values, palette="RdYlBu")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('Investor Name', fontsize=12)
        plt.ylabel('Number of Fundings Made', fontsize=12)
        plt.title("Top 10 Investors by Number of Fundings", fontsize=16)
        for i, v in enumerate(investors.values):
            plt.text(i, v + 1, str(v), ha='center', fontsize=9)
        st.pyplot(plt)

    elif selected_question == questions[9]:  # Funding Rounds by Number of Fundings
        investment = df['round'].value_counts().head(8)
        plt.figure(figsize=(15, 6))
        sns.barplot(x=investment.index, y=investment.values, palette="coolwarm")
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('Investment Type', fontsize=12)
        plt.ylabel('Number of Fundings Made', fontsize=12)
        plt.title("Funding Rounds by Number of Fundings", fontsize=16)
        for i, v in enumerate(investment.values):
            plt.text(i, v + 1, str(v), ha='center', fontsize=9)
        st.pyplot(plt)

