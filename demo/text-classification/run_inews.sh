python ../../paddlehub/commands/hub.py autofinetune text_classifier.py --param_file=autodl.yaml --gpu=0,1,2,3,4 --popsize=12 --round=10 \
 --output_dir=./autodl.output.inews.1112 --evaluator=fulltrail --tuning_strategy=hazero dataset inews
