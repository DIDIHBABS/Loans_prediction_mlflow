import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

exp_id = mlflow.create_experiment()

# mlflow.set_experiment()

with mlflow.start_run(run_name='DecisionTreeClass') as run:
    mlflow.set_tag()

mlflow.end_run()

mlflow.log_param()

mlflow.log_metric()
mlflow.set_tag()
mlflow.log_artifacts()
mlflow.log_artifacts()