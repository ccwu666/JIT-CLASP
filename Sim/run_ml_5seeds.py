import os


def ML_train_and_eval(seed, algorithm: str = None,project: str = None):
    raw_train_test = f'python sim_model.py  -algorithm {algorithm} -project {project} -seed {seed}'

    # train
    print("**********************Training and test *********************")
    os.system(raw_train_test)

if __name__ == "__main__":
    seeds = [42,88,1234,2024,2048]
    projects=["jdt","gerrit","go","openstack","platform","qt"]
    algorithms=["RandomForest"]
    for project in projects:
        for seed in seeds:
            for algorithm in algorithms:
                print(f"Running run_ml_train_and_eval model for {project},seed {seed},algorithm {algorithm}")
                ML_train_and_eval(seed, algorithm,project)