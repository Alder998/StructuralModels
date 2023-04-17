## Structural Models in forecasting Probability of Default
### Master Thesis in Quantitative Finance

Implementation of Three analytical Models and one Montecarlo Approach to compute the Probability of Default of any traded stock on Yahoo Finance.
A calibration procedure is embedded in the models, and is performed to compute the unknown terms, namely the drift and the volatility of the Assets'
Dynamics. 

It has been developed a little experiment of new Montecarlo Approach. It treates the Firms' Debt value as a pure Poisson Process, whose parameters are selected
as the regression weights of the Stock time series to the time series of some market indexes and various drivers.

The "Observed" Values of Firms' Debt and Assets' Level are taken from the library yfinance. The project has been provided with a graphical tool using
Dash. Virtually, it can show with a good accuracy, the structural PDs of any stock traded in the market.
