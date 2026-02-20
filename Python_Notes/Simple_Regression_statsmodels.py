# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3.13.11 (Conda)
#     language: python
#     name: py313
# ---

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# %% [markdown]
# # Lesson:
#
#  - ```sm.OLS``` is the array-based API --> it expects numeric matrices/arrays.
#
#  - ```ols``` (from ```statsmodels.formula.api```) is the formula-based API --> it expects a formula string, not arrays.
#
# And, ANOVA table uses the ols from ```statsmodels.formula.api```:
# https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
#
# ```result_smf.predict(X)``` from ```smf``` works just like ```result.predict(X)```. The same is true about ```conf_int()```
#
# ```get_prediction``` works like so:
#
# ```
# pred = result.get_prediction([1, x0])
# ci = pred.summary_frame(alpha=0.05)
# ```
#
# or from ```smf```:
#
# ```
# new_data = pd.DataFrame({"opp_rushing_yard": [x0]})
# pred = result_smf.get_prediction(new_data)
# ci = pred.summary_frame(alpha=0.05)
# ```

# %% [markdown]
# ## Lets stick to ```smf```
#
# ## Working with ```smf```
#
# Model like so
#
# ```
# mpg_model = smf.ols('mpg ~ engine_displacement', data = mpg_df)
# mpg_result = mpg_model.fit()
# ```
#
# - It automatically adds intercept.
# - ```mpg_result.summary()``` Produces summary that includes CI of coefficients at 95% sigfinicance level
# - ```sm.stats.anova_lm(mpg_result, typ=2)``` creates analysis-of-variance table
# - ```mpg_result.conf_int()``` shows the CI from the ```.summary()```
# - ```mpg_result.conf_int(alpha=0.01)``` shows CI at 99% significance level
# - RSS can be accessed in 2 ways:
#   - ```(mpg_result.resid ** 2).sum()```
#   - ```mpg_anova_tbl.loc["Residual", "sum_sq"]```
#   
# - To predict at new values of ```x``` we need a dataframe:
#    - new_data = pd.DataFrame({"engine_displacement": [x0]})
#    - predict_table = mpg_result.get_prediction(new_data)
#    - The above result is a table with predictions, CIs and PIs.
#    - ```yhat_tbl.summary_frame(alpha=0.01)```
#    - Predicted values are obtained by ```yhat_tbl.predicted_mean[0]```

# %% [markdown]
# ## Recall: Summary of working with ```smf```
#
# Model like so
#
# ```
# mpg_model = smf.ols('mpg ~ engine_displacement', data = mpg_df)
# mpg_result = mpg_model.fit()
# ```
#
# - It automatically adds intercept.
# - ```mpg_result.summary()``` Produces summary that includes CI of coefficients at 95% sigfinicance level
# - ```sm.stats.anova_lm(mpg_result, typ=2)``` creates analysis-of-variance table
# - ```mpg_result.conf_int()``` shows the CI from the ```.summary()```
# - ```mpg_result.conf_int(alpha=0.01)``` shows CI at 99% significance level
# - RSS can be accessed in 2 ways:
#   - ```(mpg_result.resid ** 2).sum()```
#   - ```mpg_anova_tbl.loc["Residual", "sum_sq"]```
#   
# - To predict at new values of ```x``` we need a dataframe:
#    - ```new_data = pd.DataFrame({"engine_displacement": [x0]})```
#    - ```predict_table = mpg_result.get_prediction(new_data)```
#    - The above result is a table with predictions, CIs and PIs.
#    - ```yhat_tbl.summary_frame(alpha=0.01)```
#    - Predicted values are obtained by ```yhat_tbl.predicted_mean[0]```
#    
# **See them in action [here](https://github.com/HNoorazar/Montgomery_Intro_Linear_Regression_Analysis/blob/main/CH2_SimpleLinearRegression.ipynb)**

# %% [markdown]
# # SkLearn vs statsmodel
#
# According to chatGPT: 
# - Sklearn better for ML, pipeline, production, integration. Cross-validation
# - Statsmodel good for inference.
#
#
# ```python
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# import statsmodels.formula.api as smf
#
# pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
# scores = cross_val_score(pipe, X, y, cv=5)
# pipe.fit(X, y)
# ```
#
# **statsmodel for inference**
#
# ```python
# X_transformed = pipe.named_steps["scaler"].transform(X)
# feature_names = X.columns  # or define manually
# X_df = pd.DataFrame(X_transformed, columns=feature_names)
# X_df["y"] = y
#
# formula = "y ~ " + " + ".join(feature_names)
# model = smf.ols(formula=formula, data=X_df).fit()
# print(model.summary())
# ```
#
# ------------
#
# - **robust SE**: What does that mean?: heteroskedasticity-robust standard errors
# Classical OLS assumes homoskedasticity: all errors have the same variance
# Keep the same coefficient estimates, but compute heteroskedasticity-robust standard errors
#
# ```python
# model = smf.ols(formula=formula, data=X_df).fit(cov_type="HC3")
# ```
#
# - interaction and polynomial terms:
# ```python
# model = smf.ols("y ~ x1 + x2 + x1:x2", data=X_df).fit()
# ```

# %% [markdown]
# Sklearn linear regression
#
# ```python
# linr = LinearRegression()
# linr.fit(x_train, y_train)
# ```

# %%
