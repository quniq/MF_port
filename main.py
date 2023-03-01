"""
FastAPI app for the Mutual Fund Portfolio Optimization and Execution
@author: Lucky Verma, Quniq Technologies Pvt. Ltd.
@date: 02-23-2023
@version: 1.0.0
"""

import json
import os
import logging

os.environ[
    'NUMBA_DISABLE_JIT'] = '1'  # uncomment this if you want to use pypfopt within simulation

import numpy as np
import pandas as pd
import riskfolio as rp
import vectorbt as vbt
import yfinance as yf
import quantstats as qs
import plotly.graph_objects as go
import plotly.figure_factory as ff
import riskfolio.Portfolio as pf
import riskfolio.HCPortfolio as hc
import riskfolio.RiskFunctions as rk
import riskfolio.ConstraintsFunctions as cf

from fastapi import FastAPI
from datetime import date
from datetime import timedelta
from sqlalchemy import create_engine
from timeit import default_timer as timer
from urllib.parse import quote_plus as urlquote
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from vectorbt.portfolio.nb import create_order_nb, auto_call_seq_ctx_nb
from vectorbt.portfolio.enums import SizeType, Direction

# setup loggers
logging.basicConfig(
    filename='mf-algo.fastapi.log',
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)

# get root logger
logger = logging.getLogger("mf-algoapp")

# create a file handler
handler = logging.FileHandler('mf-algo.fastapi.log')
handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

# initialize the FastAPI app
app = FastAPI()

# allow CORS
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    logger.info("logging from the root logger")
    return {"message": "Kaise hain aap log?"}


@app.get("/generate-portfolio")
async def generate_portfolio(investAmount=100_000,
                             goal=1,
                             taxSaving='no',
                             riskProfile=1,
                             startDate='2021-11-29',
                             rbRule='Y',
                             rbWindow=1260,
                             TCDollarRupee=82,
                             backtest='no'):
    """
    Function to generate the portfolio
    """
    logger.info("logging from the generate_portfolio logger")

    risk_profile = int(riskProfile)  # Low Risk 1 to High Risk 5
    TC_dollar_rupee = int(TCDollarRupee)  # 74
    invest_amount = int(investAmount)  # 1000000
    """
    Downloading the data
    """

    mf_asset_universe = pd.read_excel(
        'MF-AssetUniverseMain.xlsx',
        sheet_name="AssetUniverse")[['FundName', 'ISIN', 'Industry', 'Sector']]
    results = " - ".replace(
        '-',
        ",".join("'{}'".format(i) for i in mf_asset_universe['ISIN'].tolist()))

    try:
        test = os.listdir(os.getcwd())
        for item in test:
            if item.endswith(".pkl"):
                if item != '{0}.pkl'.format(
                        date.today()) and item != '{0}.pkl'.format(
                            date.today() - timedelta(days=1)):
                    os.remove(os.path.join(os.getcwd(), item))

        try:
            df_0 = pd.read_pickle('{0}.pkl'.format(date.today()))
        except:
            df_0 = pd.read_pickle('{0}.pkl'.format(date.today() -
                                                   timedelta(days=1)))
        df_0.set_index('Date', inplace=True)
        df_0.sort_index(inplace=True)
        print('Pickle loaded')
    except:
        # create sqlalchemy engine
        db_string = "postgresql://tsdbadmin:cqcwy6zjhtwqe2x2@hactck2q8m.ydsh56i2ps.tsdb.cloud.timescale.com:39725/tsdb?sslmode=require"
        engine = create_engine(db_string)

        sql = """
        SELECT "Date", "ISIN", "MF_Name", "Type", "NAV"
        FROM public."MUTUAL_FUND_PRICE"
        WHERE "ISIN" IN """ + "(" + results + ")"

        df_0 = pd.read_sql(sql, con=engine)

        # save the data to a pickle file
        df_0.to_pickle('{0}.pkl'.format(date.today()))
        print("DB Pickled")

    # perform data cleaning
    df_0.index = pd.to_datetime(df_0.index,
                                format='%Y-%m-%d',
                                infer_datetime_format=True)

    # df_0.sort_values(by='Date', inplace=True)

    # convert NAV to float
    df_0['NAV'] = df_0['NAV'].astype(float)

    names = mf_asset_universe['ISIN'].tolist()

    # group the df by ISIN
    grouped = df_0.groupby('ISIN')

    # select the df from grouped df
    grouped.get_group(names[0])

    data = {}

    n = len(names)
    for i in range(0, n):
        tdf = grouped.get_group(names[i])
        # df.index = df.index.str.slice(stop=10)
        tdf.index = pd.to_datetime(tdf.index,
                                   format='%Y-%m-%d',
                                   infer_datetime_format=True)
        tdf = tdf['NAV'].to_frame()
        # df = df.resample('D').last().dropna()
        tdf.columns = [names[i]]
        data[str(i)] = tdf

    data2 = pd.DataFrame([])

    for i in data.keys():
        data2 = data2.merge(data[str(i)],
                            how='outer',
                            left_index=True,
                            right_index=True)

    data3 = data2[data2.index > '2013-12-31']

    cols = data3.iloc[0, :]
    cols = cols[cols.isna() == False].index

    data3 = data3[cols].fillna(method='ffill')

    # define the mf asset universe based on the risk profile and goals

    # define the risk profile
    # 1. Conservative
    # 2. Moderate
    # 3. Balanced
    # 4. Assertive
    # 5. Aggressive

    # define the goals
    # 1. Capital Protection - capital
    # 2. Long-term Wealth Creation - longterm
    # 3. High Risk, High Reward - highrisk
    # 4. Building a Home - home
    # 5. Retirement - retirement
    # 6. Emergency Fund - emergency
    # 7. Child's education - education
    # 8. Income - income

    # define the risk profile and goals

    mf_asset_universe_1 = None
    mf_asset_universe_2 = None
    mf_asset_universe_3 = None
    mf_asset_universe_tax = None

    algo_asset_universe_1 = ""
    algo_asset_universe_2 = ""
    algo_asset_universe_3 = ""
    algo_asset_universe_tax = ""

    # conservative risk profile
    if risk_profile == 1:
        # capital protection
        if int(goal) == 1:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Gold", "Liquid", "GILT", "Long", "Dynamic&Floating"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Gold", "Liquid", "GILT", "Long", "Dynamic&Floating"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 2:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Dynamic", "Floating"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 3:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Dynamic", "Floating", "Medium", "MediumLong", "CreditRisk"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Dynamic", "Floating", "Medium", "MediumLong", "CreditRisk"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Corporate", "CreditRisk", "Dynamic", "Floating",
                    "BankingPSU"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Corporate", "CreditRisk", "Dynamic", "Floating"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 4:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["BankingPSU", "GILT", "Long", "GILTConstant"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["BankingPSU", "GILT", "Long", "GILTConstant"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "GILT", "Long", "GILTConstant", "Corporate", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Dynamic", "Floating"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 5:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Dynamic", "Floating"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Dynamic", "Floating"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "GILT", "Long", "GILTConstant", "Corporate", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Dynamic", "Floating"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 6:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Gold", "Liquid", "Dynamic", "Floating", "Short",
                    "UltraShort"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Gold", "Liquid", "Dynamic", "Floating", "Short",
                    "UltraShort"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "GILT", "Long", "GILTConstant", "Dynamic", "Floating",
                    "Low"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 7:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "GILT", "Long", "GILTConstant", "Dynamic", "Floating",
                    "Medium", "MediumLong"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "GILT", "Long", "GILTConstant", "Dynamic", "Floating",
                    "Medium", "MediumLong"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 8:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Corporate", "CreditRisk", "BankingPSU", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Corporate", "CreditRisk", "BankingPSU", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Corporate", "CreditRisk", "BankingPSU", "Dynamic",
                    "Floating"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Dynamic", "Floating"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

    # moderate risk profile
    elif risk_profile == 2:
        # capital protection
        if int(goal) == 1:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "GILT", "Long", "GILTConstant", "Dynamic", "Floating",
                    "Medium", "MediumLong"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "GILT", "Long", "GILTConstant", "Dynamic", "Floating",
                    "Medium", "MediumLong"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Corporate", "Conservative", "Medium", "MediumLong"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "Medium", "MediumLong"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 2:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Dynamic", "Floating", "BankingPSU", "Arbitrage",
                    "Conservative"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Dynamic", "Floating", "BankingPSU", "Arbitrage",
                    "Conservative"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "CreditRisk", "Medium", "MediumLong"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "CreditRisk", "Medium", "MediumLong"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 3:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiAsset", "Conservative", "CreditRisk", "Balanced"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiAsset", "Conservative", "CreditRisk", "Balanced"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Balanced", "Aggressive", "Medium", "MediumLong",
                    "Conservative"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "Balanced", "GILT", "Long", "GILTConstant"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 4:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Balanced", "Corporate", "Conservative", "Dynamic",
                    "Floating"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Balanced", "Corporate", "Conservative", "Dynamic",
                    "Floating"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "CreditRisk", "Medium", "MediumLong"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 5:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "Balanced", "GILT", "Long"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "Balanced", "GILT", "Long"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "MultiAsset", "Medium", "MediumLong"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 6:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Dynamic",
                    "Floating"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Dynamic",
                    "Floating"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Corporate", "Conservative", "Medium", "MediumLong"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 7:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiAsset", "Conservative", "Dynamic", "Floating"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiAsset", "Conservative", "Dynamic", "Floating"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "MultiAsset", "GILT", "Long",
                     "GILTConstant"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 8:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiAsset", "Conservative", "Balanced", "CreditRisk"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiAsset", "Conservative", "Balanced", "CreditRisk"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Medium", "MediumLong", "Aggressive", "Balanced",
                    "Conservative"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Balanced", "Conservative", "GILT", "Long", "GILTConstant"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

    # balanced risk profile
    elif risk_profile == 3:
        # capital protection
        if int(goal) == 1:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "LargeCap", "Balanced", "Corporate"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 2:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Balanced", "Conservative", "MultiAsset"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Balanced", "Conservative", "MultiAsset"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "LargeCap", "Aggressive", "Balanced"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 3:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Balanced", "Conservative", "Aggressive"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Balanced", "Conservative", "Aggressive"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "LargeCap", "MultiAsset", "Balanced"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Corporate", "Aggressive", "Conservative", "GILT", "Long",
                    "GILTConstant"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 4:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "LargeCap", "Balanced", "Conservative", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "LargeCap", "Balanced", "Conservative", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "LargeCap", "Conservative", "Balanced"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 5:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "Aggressive", "LargeCap", "Conservative"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "Aggressive", "LargeCap", "Conservative"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Balanced", "Aggressive", "MultiAsset"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 6:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Balanced", "Conservative", "GILT", "Long", "GILTConstant"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Balanced", "Conservative", "GILT", "Long", "GILTConstant"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Balanced", "Aggressive", "Conservative", "GILT", "Long",
                    "GILTConstant"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 7:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "LargeCap", "Balanced", "Conservative", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "LargeCap", "Balanced", "Conservative", "Medium",
                    "MediumLong"
                ])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "LargeCap", "Balanced", "Conservative"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 8:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "LargeCap", "Balanced", "MultiAsset"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "LargeCap", "Balanced", "MultiAsset"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "Large&MidCap", "LargeCap", "Balanced"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Corporate", "Aggressive", "GILT", "Long", "GILTConstant",
                    "Conservative"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

    # assertive risk profile
    elif risk_profile == 4:
        # capital protection
        if int(goal) == 1:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Balanced", "Large&MidCap", "Conservative"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Balanced", "Large&MidCap", "Conservative"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "LargeCap", "Balanced", "Large&MidCap"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "ELSS"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 2:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "Dividend", "MultiCap", "LargeCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "Dividend", "MultiCap", "LargeCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "Large&MidCap", "FlexiCap", "MultiCap"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["GILT", "Long", "GILTConstant", "ELSS"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 3:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Large&MidCap", "Aggressive", "MultiCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Large&MidCap", "Aggressive", "MultiCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["FlexiCap", "MultiCap", "Aggressive", "MidCap"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 4:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Large&MidCap", "Balanced", "Dividend"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Large&MidCap", "Balanced", "Dividend"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiCap", "LargeCap", "Large&MidCap", "Aggressive"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "ELSS", "GILT", "Long", "GILTConstant",
                    "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 5:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "Large&MidCap", "LargeCap", "FlexiCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "Large&MidCap", "LargeCap", "FlexiCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MidCap", "LargeCap", "MultiCap", "Aggressive"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "ELSS", "GILT", "Long", "GILTConstant",
                    "Balanced"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 6:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "LargeCap", "Balanced", "Large&MidCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Conservative", "LargeCap", "Balanced", "Large&MidCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Large&MidCap", "Balanced", "Aggressive"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced",
                    "ELSS"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 7:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "FlexiCap", "LargeCap", "Aggressive"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Balanced", "FlexiCap", "LargeCap", "Aggressive"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "Large&MidCap", "FlexiCap", "MultiCap"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "Conservative", "GILT", "Long", "GILTConstant", "Balanced",
                    "ELSS"
                ])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 8:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "MultiCap", "FlexiCap", "MidCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "MultiCap", "FlexiCap", "MidCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "MidCap", "Aggressive", "SmallCap"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

    # aggressive risk profile
    elif risk_profile == 5:
        # capital protection
        if int(goal) == 1:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "Large&MidCap", "FlexiCap", "MultiCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "Large&MidCap", "FlexiCap", "MultiCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "FlexiCap", "MultiCap", "Dividend"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 2:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "FlexiCap", "MultiCap", "Large&MidCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "FlexiCap", "MultiCap", "Large&MidCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["FlexiCap", "MultiCap"
                     "Large&MidCap", "MidCap"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 3:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "MidCap", "FlexiCap", "Pharma"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "MidCap", "FlexiCap", "Pharma"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "MidCap", "SmallCap", "FlexiCap", "Technology", "Thematic"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 4:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "FlexiCap", "LargeCap", "MultiCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "FlexiCap", "LargeCap", "MultiCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["MultiCap", "LargeCap", "Large&MidCap", "Dividend"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 5:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "LargeCap", "FlexiCap", "Value"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "LargeCap", "FlexiCap", "Value"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "FlexiCap", "MultiCap", "Dividend"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 6:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "LargeCap", "Aggressive", "FlexiCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "LargeCap", "Aggressive", "FlexiCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "FlexiCap", "Aggressive", "MidCap"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 7:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Large&MidCap", "FlexiCap", "Dividend"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["LargeCap", "Large&MidCap", "FlexiCap", "Dividend"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Large&MidCap", "MultiCap", "LargeCap", "Dividend"])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"

        elif int(goal) == 8:
            # define the asset universes

            mf_asset_universe_1 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["FlexiCap", "MidCap", "SmallCap", "MultiCap"])]
            algo_asset_universe_1 = "MinRisk-Classic-MV"

            mf_asset_universe_2 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["FlexiCap", "MidCap", "SmallCap", "MultiCap"])]
            algo_asset_universe_2 = "MaxRet-Classic-MV"

            mf_asset_universe_3 = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin([
                    "FlexiCap", "SmallCap", "MidCap", "Technology", "Thematic"
                ])]
            algo_asset_universe_3 = "MaxRet-Classic-MV"

            mf_asset_universe_tax = mf_asset_universe.loc[
                mf_asset_universe["Sector"].isin(
                    ["Aggressive", "ELSS", "Conservative", "Balanced"])]
            algo_asset_universe_tax = "MinRisk-Classic-MV"
    algos = [["MinRisk", "HRP", "MV"],
             algo_asset_universe_1.split("-"),
             algo_asset_universe_2.split("-"),
             algo_asset_universe_3.split("-"),
             algo_asset_universe_tax.split("-")]

    asset_universe = [
        mf_asset_universe, mf_asset_universe_1, mf_asset_universe_2,
        mf_asset_universe_3, mf_asset_universe_tax
    ]

    data_per_universe = [
        data3.loc[:,
                  data3.columns.isin(mf_asset_universe["ISIN"].to_list())],
        data3.loc[:,
                  data3.columns.isin(mf_asset_universe_1["ISIN"].to_list())],
        data3.loc[:,
                  data3.columns.isin(mf_asset_universe_2["ISIN"].to_list())],
        data3.loc[:,
                  data3.columns.isin(mf_asset_universe_3["ISIN"].to_list())],
        data3.loc[:,
                  data3.columns.isin(mf_asset_universe_tax["ISIN"].to_list())]
    ]

    portfolio_names = ["dummy", "Simple", "Classic", "Berrywise", "TaxSaving"]

    # output json holding all the portfolio values, stats, and holdings
    final_output = {}

    for _x, x in enumerate(algos):
        print("Selected algo: ", x)
        """
        Loading price data
        """

        # Price data
        Y = data_per_universe[_x].pct_change().dropna()

        # Industry & Sector data
        industry = asset_universe[_x][['ISIN', 'Industry', 'Sector']]
        print(industry)
        """
        Backtesting 
        """

        if backtest == 'yes':
            start_date = startDate
        elif backtest == 'no':
            start_date = str(date.today())
        """
        Building the backtesing object
        """

        clusters = cf.assets_clusters(returns=Y,
                                      correlation='spearman',
                                      linkage='ward',
                                      k=None,
                                      max_k=11,
                                      leaf_order=True)

        clusters = clusters.sort_index()
        clusters = pd.merge(clusters,
                            industry[['ISIN', 'Sector']],
                            left_on='Assets',
                            right_on='ISIN')

        del clusters['ISIN']

        # adjust constraints weights based on investment amount
        invest_weights = []

        if invest_amount > 50_000:
            invest_weights = [0.1, 0.25, 0.3]
        elif 50_000 >= invest_amount > 25_000:
            invest_weights = [0.15, 0.25, 0.3]
        elif 25_000 >= invest_amount > 15_000:
            invest_weights = [0.2, 0.25, 0.3]
        elif 15_000 >= invest_amount >= 10_000:
            invest_weights = [0.25, 0.25, 0.3]

        constraints = {
            'Disabled': [False, False, False],
            'Type': ['All Assets', 'All Classes', 'All Classes'],
            'Set': ['', 'Clusters', 'Sector'],
            'Position': ['', '', ''],
            'Sign': ['<=', '<=', '<='],
            'Weight': invest_weights,
            'Type Relative': ['', '', ''],
            'Relative Set': ['', '', ''],
            'Relative': ['', '', ''],
            'Factor': ['', '', '']
        }

        constraints = pd.DataFrame(constraints)

        A, B = cf.assets_constraints(constraints, clusters)

        vbt.settings.returns['year_freq'] = '252 days'

        num_tests = 2000
        ann_factor = data_per_universe[_x].vbt.returns(freq='D').ann_factor

        def prep_func_nb(simc, every_nth):
            # Define rebalancing days
            simc.active_mask[:, :] = False
            simc.active_mask[every_nth::every_nth, :] = True
            return ()

        def segment_prep_func_nb(sc, find_weights_nb, rm, model, obj,
                                 history_len, ann_factor, num_tests,
                                 srb_sharpe):
            if history_len == -1:
                # Look back at the entire time period
                close = sc.close[:sc.i, sc.from_col:sc.to_col]
            else:
                # Look back at a fixed time period
                if sc.i - history_len <= 0:
                    return (np.full(sc.group_len, np.nan),
                            )  # insufficient data
                close = sc.close[sc.i - history_len:sc.i,
                                 sc.from_col:sc.to_col]

            # Find optimal weights
            best_sharpe_ratio, weights = find_weights_nb(
                sc, rm, model, obj, close, num_tests)
            srb_sharpe[sc.i] = best_sharpe_ratio

            # Update valuation price and reorder orders
            size_type = np.full(sc.group_len, SizeType.TargetPercent)
            direction = np.full(sc.group_len, Direction.LongOnly)
            temp_float_arr = np.empty(sc.group_len, dtype=np.float_)
            for k in range(sc.group_len):
                col = sc.from_col + k
                sc.last_val_price[col] = sc.close[sc.i, col]
            auto_call_seq_ctx_nb(sc, weights, size_type, direction,
                                 temp_float_arr)

            return (weights, )

        def order_func_nb(oc, weights):
            col_i = oc.call_seq_now[oc.call_idx]
            return create_order_nb(size=weights[col_i],
                                   size_type=SizeType.TargetPercent,
                                   price=oc.close[oc.i, oc.col])

        assets = Y.columns.tolist()

        def opt_weights(sc, rm, model, obj, close, num_tests):
            # Calculate expected returns and sample covariance matrix
            close = pd.DataFrame(close, columns=assets)
            returns = close.pct_change().dropna()

            #    model = model # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            #    rm = rm # Risk measure used, this time will be variance
            #    obj = 'MinRisk' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True  # Use historical scenarios for risk measures that depend on scenarios
            rf = 0  # Risk free rate
            l = 0  # Risk aversion factor, only useful when obj is 'Utility'

            if model == 'Classic':

                # Building the portfolio object
                port = pf.Portfolio(returns=returns)

                # Select method and estimate input parameters:

                method_mu = 'hist'  # Method to estimate expected returns based on historical data.
                method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

                port.assets_stats(method_mu=method_mu,
                                  method_cov=method_cov,
                                  d=0.94)

                port.ainequality = A
                port.binequality = B

                # Calculating optimum portfolio
                #         port.solvers = ['MOSEK']

                w = port.optimization(model=model,
                                      rm=rm,
                                      obj=obj,
                                      rf=rf,
                                      l=l,
                                      hist=hist)

                # # show values greater than 0.01
                # w1 = w[w > 0.01]
                # # drop NaN values
                # w1 = w1.dropna()
                # # sort by index
                # w1 = w1.sort_index()
                # print(w1)

                weights = np.ravel(w.to_numpy())
                shp = rk.Sharpe(w,
                                port.mu,
                                cov=port.cov,
                                returns=returns,
                                rm=rm,
                                rf=0,
                                alpha=0.05)

            elif model == 'kelly':

                # Building the portfolio object
                port = pf.Portfolio(returns=returns)

                # Select method and estimate input parameters:

                method_mu = 'hist'  # Method to estimate expected returns based on historical data.
                method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

                port.assets_stats(method_mu=method_mu,
                                  method_cov=method_cov,
                                  d=0.94)

                port.ainequality = A
                port.binequality = B

                # Calculating optimum portfolio
                #         port.solvers = ['MOSEK']

                w = port.optimization(model='Classic',
                                      rm=rm,
                                      obj=obj,
                                      kelly='exact',
                                      rf=rf,
                                      l=l,
                                      hist=hist)

                weights = np.ravel(w.to_numpy())
                shp = rk.Sharpe(w,
                                port.mu,
                                cov=port.cov,
                                returns=returns,
                                rm=rm,
                                rf=0,
                                alpha=0.05)

            elif model in ['HRP', 'HERC']:

                port = hc.HCPortfolio(returns=returns)
                #        model=model # Could be HRP or HERC
                correlation = 'pearson'  # Correlation matrix used to group assets in clusters
                #        rm = rm # Risk measure used, this time will be variance
                rf = 0  # Risk free rate
                linkage = 'ward'  # Linkage method used to build clusters
                max_k = 11  # Max number of clusters used in two difference gap statistic
                leaf_order = True  # Consider optimal order of leafs in dendrogram

                w = port.optimization(model=model,
                                      correlation=correlation,
                                      rm=rm,
                                      rf=rf,
                                      linkage=linkage,
                                      max_k=max_k,
                                      leaf_order=leaf_order)

                weights = np.ravel(w.to_numpy())
                shp = rk.Sharpe(w,
                                returns.mean(),
                                cov=port.cov,
                                returns=returns,
                                rm=rm,
                                rf=0,
                                alpha=0.05)

            return shp, weights

        sharpe = {}
        portfolio = {}

        k, j, i = x[0], x[1], x[2]
        sharpe[k + "-" + j + "-" + i] = np.full(data_per_universe[_x].shape[0],
                                                np.nan)
        print(k + "-" + j + "-" + i)

        # print(data_per_universe[idx].dtypes, data_per_universe[idx].shape, type(data_per_universe[idx]))
        # print("sharpe", len(sharpe.values()))
        # print("ann_factor", ann_factor, num_tests)

        print("Optimizing weights...")

        # Run simulation with a custom order function (Numba should be disabled)
        portfolio[k + "-" + j + "-" + i] = vbt.Portfolio.from_order_func(
            data_per_universe[_x],
            order_func_nb,
            prep_func_nb=prep_func_nb,
            prep_args=(
                252, ),  # Cambiando la frecuencia de rebalanceo de 21 a 252
            segment_prep_func_nb=segment_prep_func_nb,
            segment_prep_args=(opt_weights, i, j, k, 252 * 7, ann_factor,
                               num_tests, sharpe[k + "-" + j + "-" + i]),
            cash_sharing=True,
            group_by=True,
            freq='D',
            incl_unrealized=True,
            seed=12)

        values = pd.DataFrame([])
        stats = pd.DataFrame([])
        weights = {}

        k, j, i = x[0], x[1], x[2]
        a = portfolio[k + "-" + j + "-" + i].value().iloc[252 * 8:]
        #     display(a.shape)
        b = a.pct_change().vbt.returns(freq='D').stats(0)
        w = portfolio[k + "-" + j + "-" + i].holding_value(
            group_by=False).vbt / portfolio[k + "-" + j + "-" + i].value()
        idx = np.flatnonzero(
            (portfolio[k + "-" + j + "-" + i].share_flow() != 0).any(axis=1))
        w = w.iloc[idx, :]
        values = pd.concat([values, a], axis=1, join='outer')
        stats = pd.concat([stats, b], axis=1)
        weights[k + "-" + j + "-" + i] = w

        # format the ouputs
        algo_name = [x[0] + "-" + x[1] + "-" + x[2]]

        values.columns = algo_name
        stats.columns = algo_name

        x = weights[algo_name[0]].columns.tolist()
        y = [
            float("{:.3f}".format(element * 100))
            for element in weights[algo_name[0]].iloc[-1].tolist()
        ]

        text = [['Equity', '% share']]
        for k in range(len(x)):
            text.append([x[k], y[k]])
        temp_bar_x = []
        temp_bar_y = []
        for j in text:
            if j[1] != 0:
                temp_bar_x.append(j[0])
                temp_bar_y.append(j[1])
        fig = go.Figure(data=[
            go.Bar(
                x=[i[:-3] for i in temp_bar_x[1:]],
                y=temp_bar_y[1:],
                # hovertext=['27% market share', '24% market share', '19% market share']
            )
        ])
        # fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
        #                   marker_line_width=1.5, opacity=0.6)
        # fig.show()

        import plotly.figure_factory as ff
        colorscale = [[0, '#272D31'], [.5, '#ffffff'], [1, '#ffffff']]
        # font = ['#FCFCFC', '#00EE00', '#008B00', '#004F00', '#660000', '#CD0000', '#FF3030']

        hodlings = []
        for m in text:
            if m[1] != 0:
                hodlings.append(m)

        fig = ff.create_table(hodlings, colorscale=colorscale)
        fig.layout.width = 250
        # fig.show()

        # display(weights['HC-HERC-MV']['TCS.NS'])

        pd_hodl = pd.DataFrame([])

        pd_hodl['MF'] = temp_bar_x[1:]

        pd_hodl['%share'] = temp_bar_y[1:]

        pd_hodl.sort_values(by='%share', ascending=False)

        # map the pd_hodl equity names to their respective FundName from mf table
        pd_hodl['MF'] = pd_hodl['MF'].map(
            mf_asset_universe.set_index('ISIN')['FundName'])

        # structure the output for this algo in a json format
        algo_output = {}
        algo_output['algo_name'] = algo_name[0]

        # nest the stats and values in algo_output using jsonable_encoder
        algo_output['stats'] = jsonable_encoder(stats.fillna("").to_dict())
        algo_output['values'] = jsonable_encoder(values.fillna("").to_dict())

        # nest the pd_hodl in algo_output
        algo_output['pd_hodl'] = jsonable_encoder(pd_hodl.fillna("").to_dict())

        # update the algo_output in the final output dict
        print(portfolio_names[_x])
        final_output[portfolio_names[_x]] = algo_output

    if taxSaving == 'yes':
        # return last algo output
        return final_output["TaxSaving"]

    # drop dummy algo output
    final_output.pop("dummy", None)

    return final_output
