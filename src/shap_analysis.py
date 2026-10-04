import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

df = pd.read_csv("data/ATL_features.csv", index_col="hour", parse_dates=True)

# Use the 20 features we kept
in_file = open("results/feature_trim.json", "r")
trim_info = json.load(in_file)
in_file.close()
cols = trim_info["kept_features"]

# Train on 2019-2023, explain predictions on 2024-2025
train = df[df.index.year < 2024]
test = df[df.index.year >= 2024]

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", random_state=42)
model.fit(train[cols], train["future_ifr"])

# Compute SHAP values for every test hour
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test[cols])

# Rank features by average size of their push (ignoring direction)
mean_abs = np.abs(shap_values).mean(axis=0)
ranking = pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs})
ranking = ranking.sort_values("mean_abs_shap", ascending=False)
print(ranking)

# Chart 1: bar chart of overall importance
plt.figure()
shap.summary_plot(shap_values, test[cols], plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/shap_bar.png", dpi=150)
plt.close()

# Chart 2: summary plot (one dot per hour, colored by feature value)
plt.figure()
shap.summary_plot(shap_values, test[cols], show=False)
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=150)
plt.close()

# Chart 3: how the spread feature changes the prediction
plt.figure()
shap.dependence_plot("spread", shap_values, test[cols], show=False)
plt.tight_layout()
plt.savefig("results/shap_spread.png", dpi=150)
plt.close()

print("Saved charts to results/")