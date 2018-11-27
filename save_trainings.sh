TRAIN_SESSION=$1
mkdir trainings/$TRAIN_SESSION trainings/$TRAIN_SESSION/models/ trainings/$TRAIN_SESSION/results trainings/$TRAIN_SESSION/figs trainings/$TRAIN_SESSION/images
mv models/* trainings/$TRAIN_SESSION/models/
mv results/* trainings/$TRAIN_SESSION/results/
mv figs/* trainings/$TRAIN_SESSION/figs/
mv images/* trainings/$TRAIN_SESSION/images/

echo "Input info of training"
read info
TRAIN_INFO_FILE=trainings/$TRAIN_SESSION/info.txt

touch $TRAIN_INFO_FILE && echo $info > $TRAIN_INFO_FILE