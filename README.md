# First: How to use the `convert_to_records.py` program

example usage (only compatible with python 2.x): 
```
python convert_to_records.py
    --train_directory=/Users/dgu/Documents/projects/tensorflow/rnn/data
    --validation_directory=/Users/dgu/Documents/projects/tensorflow/rnn/data
    --output_directory=/Users/dgu/Documents/projects/tensorflow/rnn/data
    --labels_file=/Users/dgu/Documents/projects/tensorflow/rnn/data/label 
```

**origin file link**: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py

**reference link** for how to write data to tfrecord and read data from tfrecord:
https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#file-formats

# Second: How to run the training program

`python -B main_train.py --data_dir=/Users/dgu/Documents/projects/tensorflow/rnn/data/sharded_data`

This command will run the training subset by default