# ImageNet-Example

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

ImageNet is a large scale visual recognition challenge run by Stanford and Princeton. The competition covers standard object classification as well as identifying object location in the image. This repo provides code for the ImageNet classification exercise.
  
**This example is currenlty in development; thus,not complete.** If you find bugs, log an issue or send a PR. Help is welcome.

---
**Model Options**

These structures have won and/or are well known from the competition. They exist under the Models directory and are implemented in CNNImageNetMain.

- AlexNet
- VGGNetA
- VGGNetD
- Google LeNet (in progress on branch)

---
**Run Program**

Current configuration provides the option to run locally, on a standalone Spark instance or on Spark cluster. After pulling down the repo and compiling the jar file, run the CNNImageNetMain class. You can pass in several arguments to change how it runs.

Command Line Arguments

    -- version <choose between Standard, SparkStandAlone, SparkCluster>
    -- modelType <choose between AlexNet, VGGNetA, VGGNetB>
    -- batchSize <integer>
    -- testBatchSize <integer>
    -- numBatches <integer>
    -- numTestBatches <integer>
    -- iterations <integer>
    -- numEpochs <integer>
    -- numCategories <integer>
    -- trainFolder <path to train file(s)>
    -- testFolder <path to test file(s)>
    -- saveModel <boolean>
    -- saveParams <boolean> 
    -- confName <path to save model>
    -- paramName <path to save params>
    
- version sets whether to run locally, on Spark locally or on a Spark cluster
- modelType sets which model to apply
- batchSize & testBatchSize sets the size of the batch for train and test
- numBatches & numTestBatches sets how many batches for train and test
- iterations sets how many times to iterate over each data batch on a layer
- numEpochs sets how many times to run the full dataset through all model layers
- numCategories sets how many categories to load for the dataset. Useful when grabbing subsets and only works on local.
- trainFolder & testFolder are the string paths where the data for each group can be found
- saveModel & saveParams set to true to save the final model configuration and the parameters (final weights)
- confName & paramName are the string paths to save the model and parameters

---
**Data**

ImageNet data can be accessed at [ImageNet site](http://image-net.org/challenges/LSVRC/2015/) with an approved account.

To work with a small sample of data, use the urls listed in the following files, download the images into the resources folder under new folders called train and test:

- [https://github.com/deeplearning4j/ImageNet-Example/blob/master/src/main/resources/image_train_urls.txt](image_train_urls.txt)
- [https://github.com/deeplearning4j/ImageNet-Example/blob/master/src/main/resources/image_test_urls.txt](image_test_urls.txt)

Note: When working with data its valuable to change up the color, shape, orientation, pixels, size, etc. There is extensive literature that can help with how to adapt data.

---
**Citation**

Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. 
(* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.

There are a number of research papers that can be found on the competition and models.
