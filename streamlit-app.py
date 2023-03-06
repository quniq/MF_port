import requests
import json
import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="Berrywise",
    page_icon=":strawberry:",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# lookup list
expReturnsRange = [
    '0', '2.40% - 6.19%', '6.19%	- 10.07%', '10.07% - 14.03%',
    '14.03% - 18.07%', '18.07% - 21.85%'
]
RiskProfiles = [
    'Conservative', 'Moderate', 'Balanced', 'Assertive', 'Aggressive'
]
volatilityRange = [
    '0', '3.35% - 4.34%', '4.34% - 6.30%', '6.30% - 8.72%', '8.72% - 11.36%',
    '11.36% - 16.81%'
]
maxDailyRange = [
    '0', '1.59% - 1.86%', '1.86% - 2.13%', '2.13% - 2.73%', '2.73 - 3.64%',
    '3.64% - 5.60%'
]
maxDrawRange = [
    '0', '9.78% - 11.21%', '11.21% - 13.15%', '13.15% - 17.63%',
    '17.63% - 23.70%', '23.70% - 33.79%'
]



riskScores = {
    'investAmount': 0,
    'age': 0,
    'duration': 0,
    'goal': 0,
    'anticipate': 0,
    'investKnowledge': 0,
    'secure': 0,
    'totalInvest': 0
}

finalRiskProfile = 0
response = None

# create session state
if 'finalRiskProfile' not in st.session_state:
    st.session_state.finalRiskProfile = finalRiskProfile
if 'finalRiskScore' not in st.session_state:
    st.session_state.finalRiskScore = 0
if 'riskScores' not in st.session_state:
    st.session_state.riskScores = riskScores
if 'response' not in st.session_state:
    st.session_state.response = response


"""
### Mutual Fund Portfolio Creation
"""

# create a sidebar
with st.sidebar:

    # name = st.text_input('NAME')
    age = st.select_slider('Age (Years)',
                            options=[
                                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 'Above 60'
                            ])
    investAmount = st.number_input('Investment Amount', value=100_000)
    duration = st.slider('Investment Duration (Years)',
                            min_value=1,
                            max_value=50)

    ### define the goals
    # 1. Capital Protection - capital
    # 2. Long-term Wealth Creation - longterm
    # 3. High Risk, High Reward - highrisk
    # 4. Building a Home - home
    # 5. Retirement - retirement
    # 6. Emergency Fund - emergency
    # 7. Child's education - education
    # 8. Income - income

    goal = st.selectbox('Investment Goal', ('Capital Protection', 'Long-term Wealth Creation', 'High Risk, High Reward', 'Building a Home', 'Retirement', 'Emergency Fund', 'Child\'s education', 'Income'))

    anticipate = st.selectbox(
        "How much you anticipate needing a portion of your invested funds before completion "
        "of "
        "the duration of your investment?",
        ('Less than 10%', 'Between 10% - 30%', 'Between 30% - 50%',
            'More than 50%'))

    investKnowledge = st.selectbox(
        "How familiar are you with investment matters?",
        ('Not familiar', 'Not very familiar', 'Somewhat familiar',
            'Fairly familiar', 'Very familiar'))

    secure = st.selectbox(
        'How secure is your current and future income from sources such as salary, pensions or '
        'other '
        'investments?',
        ('Not secure', 'Somewhat secure', 'Fairly secure', 'Very secure'))
    totalInvest = st.selectbox(
        'What is the proportion the amount you are investing respect of your total '
        'investment '
        'assets?', ('Less than 10%', 'Between 10% - 30%',
                    'Between 30% - 50%', 'More than 50%'))

    taxSaving = st.selectbox('Tax Saving?', ('yes', 'no'))

    try:
        # Age
        if age == 'Above 60':
            riskScores['age'] = 1
        elif age >= 60:
            riskScores['age'] = 1
        elif 59 >= age >= 50:
            riskScores['age'] = 3
        elif 50 >= age >= 40:
            riskScores['age'] = 5
        elif 40 >= age >= 30:
            riskScores['age'] = 7
        elif 30 >= age >= 15:
            riskScores['age'] = 9

        # Investment Amount
        if investAmount >= 10**7:
            riskScores['investAmount'] = 9
        elif 10**7 >= investAmount >= 10**7 * 0.75:
            riskScores['investAmount'] = 7
        elif 10**7 * 0.75 >= investAmount >= 10**7 * 0.5:
            riskScores['investAmount'] = 5
        elif 10**7 * 0.5 >= investAmount >= 10**7 * 0.25:
            riskScores['investAmount'] = 3
        elif 10**7 * 0.25 >= investAmount:
            riskScores['investAmount'] = 1

        # Duration of Investment
        if duration >= 25:
            riskScores['duration'] = 9
        elif 25 >= duration >= 16:
            riskScores['duration'] = 7
        elif 15 >= duration >= 11:
            riskScores['duration'] = 5
        elif 10 >= duration >= 6:
            riskScores['duration'] = 3
        elif 5 >= duration >= 1:
            riskScores['duration'] = 1

        # Goal
        if goal == 'Capital Protection':
            riskScores['goal'] = 1
        elif goal == 'Long-term Wealth Creation':
            riskScores['goal'] = 7
        elif goal == 'High Risk, High Reward':
            riskScores['goal'] = 9
        elif goal == 'Building a Home':
            riskScores['goal'] = 5
        elif goal == 'Retirement':
            riskScores['goal'] = 5
        elif goal == 'Emergency Fund':
            riskScores['goal'] = 3
        elif goal == 'Child\'s education':
            riskScores['goal'] = 5
        elif goal == 'Income':
            riskScores['goal'] = 7

        # anticipate
        if anticipate == 'Less than 10%':
            riskScores['anticipate'] = 9
        elif anticipate == 'Between 10% - 30%':
            riskScores['anticipate'] = 7
        elif anticipate == 'Between 30% - 50%':
            riskScores['anticipate'] = 3
        elif anticipate == 'More than 50%':
            riskScores['anticipate'] = 1

        # Investment Knowledge
        if investKnowledge == 'Not familiar':
            riskScores['investKnowledge'] = 1
        elif investKnowledge == 'Not very familiar':
            riskScores['investKnowledge'] = 3
        elif investKnowledge == 'Somewhat familiar':
            riskScores['investKnowledge'] = 5
        elif investKnowledge == 'Fairly familiar':
            riskScores['investKnowledge'] = 7
        elif investKnowledge == 'Very familiar':
            riskScores['investKnowledge'] = 9

        # Security
        if secure == 'Not secure':
            riskScores['secure'] = 1
        elif secure == 'Somewhat secure':
            riskScores['secure'] = 3
        elif secure == 'Fairly secure':
            riskScores['secure'] = 5
        elif secure == 'Very secure':
            riskScores['secure'] = 7

        # Total Investment Proportion
        if totalInvest == 'Less than 10%':
            riskScores['totalInvest'] = 9
        elif totalInvest == 'Between 10% - 30%':
            riskScores['totalInvest'] = 7
        elif totalInvest == 'Between 30% - 50%':
            riskScores['totalInvest'] = 3
        elif totalInvest == 'More than 50%':
            riskScores['totalInvest'] = 1


    except Exception as e:
        print(e)

    # local generated values
    # print(riskScores)
    totalRiskScore = sum(riskScores.values())

    # Risk Profile brackets
    if totalRiskScore >= 56:
        finalRiskProfile = RiskProfiles[4]
        tempMaxDraw = '23.70-33.79%'
        tempExpReturnsScore = 5
        tempMaxDrawScore = 5
    elif 56 >= totalRiskScore >= 42:
        finalRiskProfile = RiskProfiles[3]
        tempMaxDraw = '17.63-23.70%'
        tempExpReturnsScore = 4
        tempMaxDrawScore = 4
    elif 42 >= totalRiskScore >= 28:
        finalRiskProfile = RiskProfiles[2]
        tempMaxDraw = '13.15-17.63%'
        tempExpReturnsScore = 3
        tempMaxDrawScore = 3
    elif 28 >= totalRiskScore >= 14:
        finalRiskProfile = RiskProfiles[1]
        tempMaxDraw = '11.21-13.15%'
        tempExpReturnsScore = 2
        tempMaxDrawScore = 2
    elif 14 >= totalRiskScore >= 0:
        finalRiskProfile = RiskProfiles[0]
        tempMaxDraw = '9.78-11.21%'
        tempExpReturnsScore = 1
        tempMaxDrawScore = 1

    # print(totalRiskScore, finalRiskProfile)

    # # Auto Populated Section (Can be altered)
    # with st.spinner("Loading ..."):
    #     st.markdown("***")
    #     expReturns = st.slider(
    #         'Expected Returns (% p.a.) (Auto-generated)',
    #         value=int(totalRiskScore // 3.2),
    #         min_value=0,
    #         max_value=25)

    #     maxDraw = st.select_slider('Drawdown Tolerance (Auto-generated)',
    #                                 value=tempMaxDraw,
    #                                 options=[
    #                                     '9.78-11.21%', '11.21-13.15%',
    #                                     '13.15-17.63%', '17.63-23.70%',
    #                                     '23.70-33'
    #                                     '.79%'
    #                                 ])

    # Update the session state
    st.session_state['finalRiskScore'] = totalRiskScore
    st.session_state['finalRiskProfile'] = finalRiskProfile
    st.session_state['investAmount'] = investAmount
    st.session_state['goal'] = riskScores['goal']
    st.session_state['taxSaving'] = taxSaving

    # generate the portfolio
    if st.button('Generate Portfolio'):
        with st.spinner(text='Portfolio Generation & Backtesting...'):
            investAmount = st.session_state['investAmount']
            goal = st.session_state['goal']
            taxSaving = st.session_state['taxSaving']
            riskProfile = st.session_state['finalRiskProfile']

            if riskProfile == "Conservative":
                riskProfile = 1
            elif riskProfile == "Moderate":
                riskProfile = 2
            elif riskProfile == "Balanced":
                riskProfile = 3
            elif riskProfile == "Aggressive":
                riskProfile = 4
            elif riskProfile == "High Risk":
                riskProfile = 5

            # # show all session state values
            # st.write('All session values:')
            # for key, value in st.session_state.items():
            #     st.write(f'{key}: {value}')

            # print all parameters passed to the function
            st.success(f"Generating Portfolio for {investAmount} and tax saving: {taxSaving}")

            #### Portfolio Generation using API call

            # curl -X 'POST' \
            # 'http://3.23.46.245:9000/generate-portfolio?investAmount=100000&goal=1&taxSaving=no&riskProfile=1&startDate=2021-11-29&rbRule=Y&rbWindow=1260&TCDollarRupee=82&backtest=no' \
            # -H 'accept: application/json' \
            # -d ''

            # API call
            url = "http://3.23.46.245:9000/generate-portfolio"
            querystring = {
                "investAmount": investAmount,
                "goal": goal,
                "taxSaving": taxSaving,
                "riskProfile": riskProfile,
                "startDate": "2021-11-29",
                "rbRule": "Y",
                "rbWindow": "1260",
                "TCDollarRupee": "82",
                "backtest": "no"
            }

            response = requests.request("POST", url, params=querystring)

            # save the response in session state
            st.session_state['response'] = response.json()

    # if clear button is clicked clear the session state
    if st.button('Clear'):
        st.session_state.clear()



# Portfolio Generation

# check is response is present in session state
if st.session_state.get('finalRiskProfile', None) is None:
    st.warning("Please answer all questions to generate your portfolio")

if st.session_state.get('response', None) is not None:
    results = st.session_state['response']

    # st.json(results)

    if taxSaving == 'yes':
        # show stats for the portfolio
        st.markdown('## Portfolio Stats')
        st.table(results['stats'])
        # convert the values to dataframe
        values_df = pd.DataFrame(results['values'])
        # convert the dates to datetime
        values_df['date'] = pd.to_datetime(values_df.index)


        # plot the values_df using plotly
        st.markdown('## Portfolio Values Plot')
        fig = px.line(values_df, x='date', y=values_df.columns, title='Portfolio Values')
        st.plotly_chart(fig, use_container_width=True)


        # show the weights
        st.markdown('## Portfolio Weights')
        st.table(results['pd_hodl'])


        # # show the df
        # st.markdown('## Portfolio Values')
        # st.table(values_df.head())

    else:            
        results_1 = results['Simple']
        values_df1 = pd.DataFrame(results_1['values'])
        values_df1['date'] = pd.to_datetime(values_df1.index)

        results_2 = results['Classic']
        values_df2 = pd.DataFrame(results_2['values'])
        values_df2['date'] = pd.to_datetime(values_df2.index)

        results_3 = results['Berrywise']
        values_df3 = pd.DataFrame(results_3['values'])
        values_df3['date'] = pd.to_datetime(values_df3.index)



        # show the plots for all three portfolios in one plot
        st.markdown('## All Portfolios')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=values_df1['date'], y=values_df1[values_df1.columns[0]].to_list(), name='Simple'))
        fig.add_trace(go.Scatter(x=values_df2['date'], y=values_df2[values_df2.columns[0]].to_list(), name='Classic'))
        fig.add_trace(go.Scatter(x=values_df3['date'], y=values_df3[values_df3.columns[0]].to_list(), name='Berrywise'))
        st.plotly_chart(fig, use_container_width=True)
        
        
        # Three columns for the three portfolios
        col1, col2, col3 = st.columns(3)

        with col1:
            # # plot the values_df using plotly
            # st.markdown('## Portfolio Values Plot')
            # fig = px.line(values_df1, x='date', y=values_df1.columns, title='Portfolio Values')
            # st.plotly_chart(fig, use_container_width=True)

            # show the stats
            st.markdown('## Portfolio Stats')
            st.table(results_1['stats'])

            # show the weights
            st.markdown('## Portfolio Weights')
            st.table(results_1['pd_hodl'])

        with col2:
            # # plot the values_df using plotly
            # st.markdown('## Portfolio Values Plot')
            # fig = px.line(values_df2, x='date', y=values_df2.columns, title='Portfolio Values')
            # st.plotly_chart(fig, use_container_width=True)

            # show the stats
            st.markdown('## Portfolio Stats')
            st.table(results_2['stats'])

            # show the weights
            st.markdown('## Portfolio Weights')
            st.table(results_2['pd_hodl'])
        
        with col3:
            # # plot the values_df using plotly
            # st.markdown('## Portfolio Values Plot')
            # fig = px.line(values_df3, x='date', y=values_df3.columns, title='Portfolio Values')
            # st.plotly_chart(fig, use_container_width=True)

            # show the stats
            st.markdown('## Portfolio Stats')
            st.table(results_3['stats'])

            # show the weights
            st.markdown('## Portfolio Weights')
            st.table(results_3['pd_hodl'])

