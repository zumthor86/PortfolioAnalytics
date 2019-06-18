from bokeh.embed import file_html
from bokeh.resources import CDN
from flask import Flask, render_template, request, jsonify
from portfolio_help.helpers import PORTFOLIOS, \
    create_sector_dictionary, \
    check_portfolio_name, \
    create_portfolio_analytics, \
    persist_portfolio, \
    calculate_common_returns, \
    calculate_cumulative_returns, \
    create_figure_line, \
    calculate_portfolios_profile, retrieve_weights

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', portfolio_options=PORTFOLIOS)


@app.route('/compare', methods=['GET'])
def compare():
    if request.method == "GET":
        return render_template('compare.html', portfolio_options=PORTFOLIOS)


@app.route('/construct', methods=['GET', 'POST'])
def construct():
    shares = create_sector_dictionary()

    groups = list(shares.keys())

    if request.method == "GET":
        return render_template('construct.html', shares=shares, groups=groups)

    if request.method == "POST":
        persist_portfolio(portfolio_name=request.form.get("portfolio_name"),
                          constituent_names=request.form.getlist("shares"),
                          weighting_scheme=request.form.get("weighting_scheme"))

        return render_template('construct.html', shares=shares)


@app.route("/check_name", methods=['GET'])
def check_name():
    portfolio_name = request.args.get("portfolio_name", 0, type=str)

    return jsonify(check_portfolio_name(portfolio_name))


@app.route("/update_analytics", methods=['GET'])
def update_analytics():
    shares = request.args.getlist("shares[]")

    p = create_portfolio_analytics(shares)

    html = file_html(p, CDN)

    return html


@app.route("/update_portfolio_analytics", methods=['GET'])
def update_portfolio_analytics():
    portfolios_selected = request.args.getlist("portfolios[]")

    common_returns = calculate_common_returns(portfolios_selected)

    portfolios_profiles_dict = calculate_portfolios_profile(common_returns)

    cumulative_returns = common_returns.apply(calculate_cumulative_returns)

    p = create_figure_line(cumulative_returns)

    html = file_html(p, CDN)

    return render_template('portfolio_analytics.html',
                           html=html,
                           portfolios_profiles=portfolios_profiles_dict,
                           portfolio_measures=list(portfolios_profiles_dict.keys()),
                           portfolios_selected=portfolios_selected)


@app.route("/update_table", methods=['GET'])
def update_table():
    portfolio = request.args.get('portfolio')

    constituents = retrieve_weights(portfolio)

    return render_template('portfolio_table.html', constituents=constituents)


if __name__ == '__main__':
    app.run()
