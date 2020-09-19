#!/bin/sh

START=$1
END=$2
ID=$START
NUM_THREADS=$3
NAME=GL_min_bayes4.py

while [ $ID -lt $END ]; do

  if [ $(( $RUNNING_PROCESSES )) -lt $(( $NUM_THREADS )) ]
  then

    echo about to run with ID=$ID

    python3 $NAME $ID &

    ID=$(( $ID + 1 ))

    echo ID : $ID

  fi
  sleep 1

  RUNNING_PROCESSES=$(ps -ef | grep $NAME | egrep -v "grep|vi|more|pg" | wc -l)

  echo RUNNING_PROCESSES : $RUNNING_PROCESSES
done
