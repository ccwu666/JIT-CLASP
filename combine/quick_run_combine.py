import os


def combine(seed, sim_method: str = None,com_method: str = None,project: str = None):
    com_path = "./com_pre_processed/original_jitclasp/attention" + "_" + project + "_" + str(seed) + ".csv"
    sim_path = "./sim_pre/pred_scores/" + project + "_" + sim_method + "_" + str(seed) + ".csv"
    raw_train_test = f'python combination.py  -sim_method {sim_method} -com_method {com_method} -project {project} -seed {seed} -com_path={com_path} -sim_path={sim_path}'
    os.system(raw_train_test)


if __name__ == "__main__":
    seeds = [42,88,1234,2024,2048]
    projects=["gerrit","go","jdt","openstack","platform","qt"]
    sim_methods=["RandomForest"]
    com_methods=["original_jitclasp"]
    for project in projects:
        for seed in seeds:
            for sim_method in sim_methods:
                for com_method in com_methods:
                    combine(seed,sim_method, com_method,project)