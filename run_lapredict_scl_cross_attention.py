import os


def X_scl_cross_attention_train_and_eval(seed, project: str = None):
    raw_train = f'python -m JIT-CLASP.concat.run \
    --output_dir=model/scl_cross/{project}/{seed}/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/lapredict_unfied/{project}/changes_train.pkl data/lapredict_unfied/{project}/features_train.pkl \
    --eval_data_file data/lapredict_unfied/{project}/changes_valid.pkl data/lapredict_unfied/{project}/features_valid.pkl\
    --test_data_file data/lapredict_unfied/{project}/changes_test.pkl data/lapredict_unfied/{project}/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 128 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --feature_size 14 \
    --patience 5 --alpha 0.5 --temp 0.1 \
    --seed {seed} 2>&1| tee model/scl_cross/{project}/{seed}/saved_models_concat/train.log'

    raw_test = f'python -m JIT-CLASP.concat.run \
    --output_dir=model/scl_cross/{project}/{seed}/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/lapredict_unfied/{project}/changes_train.pkl data/lapredict_unfied/{project}/features_train.pkl \
    --eval_data_file data/lapredict_unfied/{project}/changes_valid.pkl data/lapredict_unfied/{project}/features_valid.pkl\
    --test_data_file data/lapredict_unfied/{project}/changes_test.pkl data/lapredict_unfied/{project}/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 256 \
    --eval_batch_size 25 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --only_adds --alpha 0.5 --temp 0.1\
    --seed {seed} 2>&1 | tee model/scl_cross/{project}/{seed}/saved_models_concat/test.log'

    # train
    print("**********************Training*********************")
    os.system(raw_train)

    print("**********************Evaluating*********************")
    os.system(raw_test)


if __name__ == "__main__":
    seeds = [42,88,1234,2024,2048]

    projects=["jdt","gerrit","go","openstack","platform","qt"]
    # projects = ["jdt"]
    for project in projects:
        for seed in seeds:
            print(f"Running SBCL_cross_attetion_train_and_eval model for {project},seed {seed}")
            X_scl_cross_attention_train_and_eval(seed, project)
