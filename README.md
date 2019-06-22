# Portfolio Analytics

## Introduction
This app brings together finance and data visualisation in a way that allows users to interactively build and compare
portfolios of their choosing. 

## Summary of dependencies
* Web framework/Templating : Flask + jinja2
* Styling : Bootstrap4
* Data visualization : Bokeh
* Data manipulation : pandas
* Website interactions/ajax : jquery
* Javascript libraries : Select2, DataTable
* Portfolio optimization : [pyportfoliopot](https://github.com/robertmartin8/PyPortfolioOpt)
* Data : [alphavantage](https://www.alphavantage.co/)

The app is served using python and flask/jinja2 and for data visualization the Bokeh library was used to create interactive charts. 
Interactions were driven using ajax instead of using Bokeh's own widgets, this made it easier to maintain the look and feel of the app.
Heavy usage of the pandas library was used for calculating returns, portfolio metrics etc. A simple parquet file was used for persisting returns, allowing fast retrieval of historical returns for a selection of stocks.

## Summary of functionality
### Browse
The makeup of already created portfolios can be inspected here using a datatable that permits sorting and searching.
### Compare
Historical cumulative performance can be compared over a common time frame for up to 4 portfolios. A concise summary of the portfolio metrics is provided. 
### Construct
Most of the functionality of the site falls within this page. After assigning a portfolio name a user can choose between two weighting schemes, simple equal weighting and and a mean/variance optimized portfolio.
Prior to finalizing and saving their selection of stocks users can analyze a correlation heatmap, a plot displaying the betas of each stock, as well as a sector breakdown.

To run the app obtain a alphavantage api key and store it in `config.json` within `/portfolio_help` as shown below

{
  "dev": {
    "alpha_vantage_api_key" : "<your api key>"
  }
}
