import warnings
import sys
import os

import pandas as pd
import numpy as np

import mlflow
import mlflow.pyfunc

import cloudpickle

import prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class prophetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        from prophet import Prophet

        return

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input["periods"][0])
        return self.model.predict(future)


conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "prophet={}".format(prophet.__version__),
        "cloudpickle={}".format(cloudpickle.__version__),
    ],
    "name": "fbp_env",
}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join('..','data', 'mldata.csv')
    )
    rolling_window = 0.5#float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    # Read the csv file from the URL
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        m = Prophet()
        m.fit(df)

        # Evaluate Metrics
        df_cv = cross_validation(m, initial="730 days", period="180 days", horizon="365 days")
        df_p = performance_metrics(df_cv, rolling_window=rolling_window)

        # Print out metrics
        print("Prophet model (rolling_window=%f):" % (rolling_window))
        print("  CV: \n%s" % df_cv.head())
        print("  Perf: \n%s" % df_p.head())

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("rolling_window", rolling_window)
        mlflow.log_metric("rmse", df_p.loc[0, "rmse"])

        mlflow.pyfunc.log_model("model", conda_env=conda_env, python_model=prophetWrapper(m))
        print(
            "Logged model with URI: runs:/{run_id}/model".format(
                run_id=mlflow.active_run().info.run_id
            )
        )