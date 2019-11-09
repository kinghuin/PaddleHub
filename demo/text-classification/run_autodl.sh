dataset_name=inews

python ../../paddlehub/commands/hub.py autofinetune text_classifier.py --param_file=autodl.yaml --gpu=5 --popsize=3 --round=10 \
 --output_dir=./${dataset_name}_autodl_output --evaluator=fulltrail --tuning_strategy=hazero dataset ${dataset_name}
