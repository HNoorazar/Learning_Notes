# SKlearn vs statsmodel
"""
According to chatGPT: Sklearn better for
ML, pipeline, production, integration. Cross-validation

Statsmodel good for inference.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import statsmodels.formula.api as smf

pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

scores = cross_val_score(pipe, X, y, cv=5)
pipe.fit(X, y)


###
### stats model for inference
###
X_transformed = pipe.named_steps["scaler"].transform(X)

feature_names = X.columns  # or define manually

X_df = pd.DataFrame(X_transformed, columns=feature_names)
X_df["y"] = y

formula = "y ~ " + " + ".join(feature_names)
model = smf.ols(formula=formula, data=X_df).fit()
print(model.summary())

# robust SEs: What does that mean?: heteroskedasticity-robust standard errors
# Classical OLS assumes homoskedasticity: all errors have the same variance
# Keep the same coefficient estimates, but compute heteroskedasticity-robust standard errors
model = smf.ols(formula=formula, data=X_df).fit(cov_type="HC3")

# interaction and polynomial terms:
model = smf.ols("y ~ x1 + x2 + x1:x2", data=X_df).fit()
