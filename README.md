# Radio-Signal-Classification-CNN

CNN implementation to classify Radio Signals from Outer Space.

## Data
- Training examples= 3200
- Test examples= 800
- Classes= 4 ("squiggle", "narrowband", "noise", "narrowbanddrd")

## Model Summary
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 64, 128, 32)       832       
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 128, 32)       128       
_________________________________________________________________
activation (Activation)      (None, 64, 128, 32)       0         
_________________________________________________________________
dropout (Dropout)            (None, 64, 128, 32)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 128, 64)       51264     
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 128, 64)       256       
_________________________________________________________________
activation_1 (Activation)    (None, 64, 128, 64)       0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 32, 64, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 64, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 131072)            0         
_________________________________________________________________
dense (Dense)                (None, 256)               33554688  
_________________________________________________________________
batch_normalization_2 (Batch (None, 256)               1024      
_________________________________________________________________
activation_2 (Activation)    (None, 256)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 1028      
=================================================================
Total params: 33,609,220
Trainable params: 33,608,516
Non-trainable params: 704

## Evaluation Metrics:

              precision    recall  f1-score   support

           0       0.96      0.99      0.98       200
           1       0.48      0.34      0.40       200
           2       0.49      0.61      0.54       200
           3       1.00      1.00      1.00       200

    accuracy                           0.74       800
   macro avg       0.73      0.74      0.73       800
weighted avg       0.73      0.74      0.73       800

Here: 
    - 0="squiggle" 
    - 1="narrowband"
    - 2="noise"
    - 3="narrowbanddrd"

