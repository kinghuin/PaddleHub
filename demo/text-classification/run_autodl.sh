dataset_name=xnli_zh

python ../../paddlehub/commands/hub.py autofinetune text_classifier.py --param_file=autodl.yaml --gpu=['1','2','3','4'] --popsize=4 --round=10 \
 --output_dir=./${dataset_name}_autodl_output --evaluator=fulltrail --tuning_strategy=hazero dataset dataset_name
