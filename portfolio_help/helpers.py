from os.path import dirname, join
from io import StringIO
from pandas import merge, read_parquet, read_csv, DataFrame
import requests
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker
from bokeh.plotting import figure
from bokeh.layouts import row, column
import numpy as np
from functools import reduce, partial
import json
from scipy.stats import skew, kurtosis
from bokeh.palettes import Viridis256, viridis
from bokeh.core.properties import value
import scipy.optimize as sco
from functools import lru_cache
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import statsmodels.api as sm
from bokeh.models import Span
import re

with open(join(dirname(__file__), 'data', 'config.json'), 'r') as f:
    config = json.load(f)

API_KEY = config['dev']['alpha_vantage_api_key']
RETURNS_PATH = join(dirname(__file__), 'data', 'returns.SNAPPY')
PORTFOLIO_PATH = join(dirname(__file__), 'data', 'portfolios.json')
SHARES_PATH = join(dirname(__file__), 'data', 'sp500_weightings.csv')
COMPANY_LOOKUP_PATH = join(dirname(__file__), 'data', 'company_lookup.json')

with open(PORTFOLIO_PATH, 'r') as file:
    PORTFOLIOS = json.load(file)

with open(COMPANY_LOOKUP_PATH, 'r') as file:
    COMPANIES = json.load(file)

SHARES = read_csv(SHARES_PATH, index_col=['Company'])


@lru_cache()
def request_returns(symbol):
    returns_df = read_parquet(RETURNS_PATH)

    if symbol not in returns_df.columns:
        response = requests.get('https://www.alphavantage.co/query', params={"function": "TIME_SERIES_DAILY_ADJUSTED",
                                                                             "symbol": symbol, "outputsize": "full",
                                                                             "apikey": API_KEY, "datatype": "csv"})

        if response.status_code == 200:

            symbol_df = read_csv(StringIO(response.text), parse_dates=True, index_col=0)

            returns_symbol = symbol_df[["adjusted_close"]] \
                .sort_index() \
                .pct_change() \
                .dropna() \
                .rename(columns={"adjusted_close": symbol})

            persist_returns(returns_symbol)

        else:
            return None

    else:
        returns_symbol = read_parquet(RETURNS_PATH, columns=[symbol])

    return returns_symbol


def persist_returns(df):
    returns = read_parquet(RETURNS_PATH)

    returns = returns.merge(df, on='timestamp', how='outer')

    returns.to_parquet(RETURNS_PATH)


def calculate_returns(constituent_symbols, constituent_weights):
    constituent_returns = read_parquet(RETURNS_PATH, columns=constituent_symbols)

    weighted_returns = constituent_returns \
        .dropna(how='any') \
        .mul(constituent_weights, axis='columns')

    portfolio_returns = weighted_returns.sum(axis=1).to_frame()

    return portfolio_returns


def calculate_annual_return(returns, n_periods_in_year=252):
    base = reduce(lambda x, y: x * y, 1 + returns)

    n_returns = len(returns)

    annual_return = (base ** (n_periods_in_year / n_returns)) - 1

    return annual_return


def calculate_annual_stddev(returns, n_periods_in_year=252):
    sigma_annual = returns.std() * np.sqrt(n_periods_in_year)

    return sigma_annual


def calculate_negative_sharpe(constituent_weights, constituent_symbols, risk_free_rate=0):
    returns = calculate_returns(constituent_symbols, constituent_weights)

    sharpe_ratio = calculate_annual_sharpe(returns, risk_free_rate)

    return -sharpe_ratio


def calculate_annual_sharpe(returns, risk_free_rate=0):
    sharpe_ratio = (calculate_annual_return(returns) - risk_free_rate) / calculate_annual_stddev(returns)

    return sharpe_ratio


def calculate_cumulative_returns(returns, starting_capital=1):
    cumulative_returns = ((1 + returns).cumprod() - 1) * starting_capital

    return cumulative_returns


def calculate_maximum_drawdown(cumulative_returns):
    start_index = np.argmax(np.maximum.accumulate(cumulative_returns) - cumulative_returns)

    end_index = np.argmax(cumulative_returns[:start_index])

    max_drawdown = cumulative_returns[start_index] - cumulative_returns[end_index]

    return max_drawdown


def persist_portfolio(portfolio_name, constituent_names, weighting_scheme):
    constituent_symbols = lookup_symbols(constituent_names)

    if weighting_scheme == 'optimized':

        portfolio_weights = calculate_optimized_weights_p(constituent_symbols)

    else:

        n_shares = len(constituent_symbols)

        equal_weights = n_shares * [1 / n_shares]

        portfolio_weights = {symbol: weight for symbol, weight in zip(constituent_symbols, equal_weights)}

    with open(PORTFOLIO_PATH, 'w') as portfolio_file:

        PORTFOLIOS[portfolio_name] = portfolio_weights

        json.dump(PORTFOLIOS, portfolio_file)


def lookup_symbols(constituent_names):
    constituent_symbols = SHARES.loc[constituent_names, 'Symbol'].to_list()
    return constituent_symbols


def calculate_common_returns(portfolio_names):
    all_portfolio_returns = [
        calculate_returns(list(PORTFOLIOS[portfolio_name].keys()), list(PORTFOLIOS[portfolio_name].values())) for
        portfolio_name in portfolio_names]

    common_index_returns = reduce(merge_on_timestamp, all_portfolio_returns)

    common_index_returns.columns = portfolio_names

    return common_index_returns


def calculate_portfolios_profile(common_returns):
    measure_map = {"calculate_annual_return": "Annualized Returns (%)",
                   "calculate_annual_stddev": "Annualized Volatility (%)",
                   "calculate_annual_sharpe": "Sharpe Ratio",
                   "calculate_maximum_drawdown": "Max Drawdown (%)",
                   "skew": "Skewness",
                   "kurtosis": "Kurtosis"}

    portfolios_profiles = common_returns.agg([calculate_annual_return,
                                              calculate_annual_stddev,
                                              calculate_annual_sharpe,
                                              calculate_maximum_drawdown,
                                              'skew',
                                              'kurtosis'])

    portfolio_measures = [measure_map[label] for label in portfolios_profiles.index.to_list()]

    portfolios_profiles.index = portfolio_measures

    portfolios_profiles_dict = portfolios_profiles.round(2).transpose().to_dict()

    return portfolios_profiles_dict


def calculate_optimized_weights(constituent_symbols):
    constituent_returns = read_parquet(RETURNS_PATH, columns=constituent_symbols)

    n_shares = constituent_returns.shape[1]

    args = (constituent_symbols,)

    bounds = ((0.0, 1.0),) * n_shares

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})

    result = sco.minimize(calculate_negative_sharpe, x0=n_shares * [1 / n_shares, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    weights = {symbol: weight for symbol, weight in zip(constituent_symbols, result.x)}

    return weights


def check_portfolio_name(portfolio_name):
    if portfolio_name in PORTFOLIOS.keys() or re.search('[^A-Za-z0-9_]', portfolio_name):
        return True
    else:
        return False


def create_figure_line(cumulative_returns):
    p = figure(x_axis_label='Date',
               y_axis_label='Cumulative return(%)',
               x_axis_type='datetime',
               y_axis_location='right',
               sizing_mode='scale_width',
               toolbar_location='left')

    column_names = cumulative_returns.columns.to_list()

    cumulative_returns = cumulative_returns.mul(100)

    cumulative_returns.reset_index(inplace=True)

    source = ColumnDataSource(cumulative_returns)

    n_portfolios = cumulative_returns.shape[1]

    palette = iter(viridis(n_portfolios))

    for column in column_names:
        p.line(x='timestamp', y=column, source=source, line_width=2, color=next(palette), legend=value(column))

    p.legend.location = "top_left"

    p.legend.click_policy = "hide"

    tooltips = [(column, "@" + column + '{0.00}') for column in column_names]

    tooltips.append(("Date", "@timestamp{%F}"))

    hover = HoverTool(tooltips=tooltips, mode='mouse', formatters={"timestamp": "datetime"})

    p.add_tools(hover)

    return p


def create_portfolio_analytics(constituents):
    symbols = lookup_symbols(constituents)

    constituent_correlation = calculate_correlation(symbols)

    p_heat = create_heatmap(constituent_correlation, symbols)

    p_barchart = create_sector_breakdown(constituents)

    p_betas = create_beta_dotplot(constituents)

    p_betas.min_border_top = 65

    p_betas.y_range = p_heat.y_range

    return row(column(p_heat, p_barchart), p_betas)


def create_heatmap(constituent_correlation, symbols):
    mapper = LinearColorMapper(palette=Viridis256, low=-1, high=1)
    n_shares = len(symbols)
    p = figure(title="Constituent Correlation: " + f'{n_shares} shares selected',
               tooltips=[('Share 1', '@company_1'), ('Share 2', '@company_2'), ('Correlation', '@correlation{0.00}')],
               x_axis_location="above",
               toolbar_location="below",
               x_range=constituent_correlation["share_1"].unique().tolist(),
               y_range=constituent_correlation["share_2"].unique().tolist())
    p.rect(x='share_1', y='share_2', source=constituent_correlation,
           width=1, height=1,
           fill_color={'field': 'correlation', 'transform': mapper},
           line_color=None)
    p.xaxis.major_label_orientation = 'vertical'
    color_bar = ColorBar(color_mapper=mapper,
                         ticker=BasicTicker(desired_num_ticks=10),
                         label_standoff=6, location=(0, 0))
    p.add_layout(color_bar, place='right')
    return p


def calculate_correlation(constituents):
    constituent_correlation = reduce(merge_on_timestamp, [request_returns(constituent) for constituent in constituents]) \
        .dropna(how='any') \
        .corr() \
        .reset_index() \
        .melt(id_vars='index') \
        .rename(columns={"index": "share_1", "variable": "share_2", "value": "correlation"}) \
        .round(2)

    constituent_correlation['company_1'] = list(map(lambda symbol: COMPANIES[symbol], constituent_correlation.share_1))
    constituent_correlation['company_2'] = list(map(lambda symbol: COMPANIES[symbol], constituent_correlation.share_2))
    return constituent_correlation


def retrieve_weights(portfolio_name):
    symbols = list(PORTFOLIOS[portfolio_name].keys())

    weights = list(PORTFOLIOS[portfolio_name].values())

    df = DataFrame.from_dict({'Symbol': symbols, 'Weight': weights})

    df['Weight'] = df['Weight'].mul(100).round(2)

    shares = SHARES.loc[:, ['Symbol', 'Sector', 'Beta']].reset_index()

    shares_data = shares.merge(df)

    shares_data['Beta'] = shares_data['Beta'].round(2)

    return list(shares_data.itertuples(index=False))


def create_sector_breakdown(constituents):
    symbols = SHARES.loc[constituents, :]

    sector_count = symbols.groupby('Sector')['Symbol'].agg('count')

    counts = sector_count.to_list()

    sectors = sector_count.index.to_list()

    colors = viridis(len(counts))

    source = ColumnDataSource(data={'counts': counts,
                                    'sectors': sectors,
                                    'colors': colors})

    p = figure(x_range=sectors, title="Sector Breakdown",
               toolbar_location=None, tools="", tooltips=[('Sector', '@sectors'), ('Number of Shares', '@counts')])

    p.vbar(x='sectors', top='counts', color='colors', legend=None, source=source, width=1)

    p.xaxis.major_label_orientation = 45

    return p


def create_sector_dictionary():
    shares_dict = SHARES.reset_index().groupby('Sector')['Company'].apply(list).to_dict()

    return shares_dict


def calculate_optimized_weights_p(constituent_symbols):
    constituent_returns = read_parquet(RETURNS_PATH, columns=constituent_symbols)

    mu = constituent_returns.agg('mean')

    cov = constituent_returns.cov()

    ef = EfficientFrontier(mu, cov, gamma=1)

    weights = ef.max_sharpe()

    return weights


def calculate_betas(symbol):
    returns = read_parquet(RETURNS_PATH)

    common_returns = returns.loc[:, ['^GSPC', symbol]].dropna(how='any').tail(252)

    ols_model = sm.OLS(common_returns[symbol], common_returns['^GSPC']).fit()

    return ols_model.params['^GSPC']


def create_beta_dotplot(constituents):
    shares = SHARES.loc[constituents, ['Symbol', 'Beta', 'Sector']].reset_index()

    hover_tool = HoverTool(tooltips=[('Company', '@Company'), ('Beta', '@Beta'), ('Sector', '@Sector')], names=['dot'])

    tools = [hover_tool]

    x_range_max = max(shares['Beta'])*1.1

    p = figure(title='Constituent Betas (252 trailing days)', x_range=[0, x_range_max], y_range=shares['Symbol'],
               toolbar_location=None, tools=tools)

    source = ColumnDataSource(shares)

    palette = viridis(6)

    market_beta = Span(location=1, dimension='height', line_color='grey', line_dash='dashed', line_width=2)

    p.add_layout(market_beta)

    p.segment(x0=0, y0='Symbol', x1='Beta', y1='Symbol', source=source, line_width=3, line_color=palette[1])

    p.circle(x='Beta', y='Symbol', size=15, fill_color=palette[5], line_color=palette[3], line_width=3, source=source,
             name='dot')

    return p


merge_on_timestamp = partial(merge, on='timestamp')
