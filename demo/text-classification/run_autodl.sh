dataset_name=xnli_zh

hub autofinetune text_classifier.py --param_file=autodl.yaml --cuda=['2','3','4'] --popsize=3 --round=10 \
 --output_dir=./${dataset_name}_autodl_output --evaluate_choice=fulltrail --tuning_strategy=hazero dataset dataset_name
