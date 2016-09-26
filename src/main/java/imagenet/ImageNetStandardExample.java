package imagenet;


import imagenet.Utils.DataModeEnum;
import imagenet.Utils.ImageNetDataSetIterator;


import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;


/**
 * Standard configuration used to run ImageNet on a single machine.
 */
public class ImageNetStandardExample extends ImageNetMain {

    private static final Logger log = LoggerFactory.getLogger(ImageNetStandardExample.class);

    public ImageNetStandardExample() {
    }

    public void initialize() throws Exception {
        // Build
        buildModel();
        setListeners();

        // Train
        MultipleEpochsIterator trainIter = null;
        ImageTransform flipTransform = new FlipImageTransform(new Random(42));
        ImageTransform warpTransform = new WarpImageTransform(new Random(42), 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});

        for(ImageTransform transform: transforms) {
            log.info("Training with " + (transform == null? "no": transform.toString()) + " transform");
            trainIter = loadData(numTrainExamples, transform, DataModeEnum.CLS_TRAIN);
            trainModel(trainIter);
        }

        // Evaluation
        numEpochs = 1;
        MultipleEpochsIterator testIter = loadData(numTestExamples, null, DataModeEnum.CLS_TEST);
        evaluatePerformance(testIter);

        // Save
        saveAndPrintResults();

    }

    private MultipleEpochsIterator loadData(int numExamples, ImageTransform transform, DataModeEnum dataModeEnum){
        System.out.println("Load data....");

        // TODO incorporate some formate of below code when using full validation set to pass valLabelMap through iterator
//                RecordReader testRecordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath, valLabelMap); // use when pulling from main val for all labels
//                testRecordReader.initialize(new LimitFileSplit(new File(testData), allForms, totalNumExamples, numCategories, Pattern.quote("_"), 0, new Random(123)));

        return new MultipleEpochsIterator(numEpochs,
                new ImageNetDataSetIterator(batchSize, numExamples,
                        new int[] {HEIGHT, WIDTH, CHANNELS}, numLabels, maxExamples2Label, dataModeEnum, splitTrainTest, transform, normalizeValue, rng), asynQues);
    }


    private void trainModel(MultipleEpochsIterator data){
        System.out.println("Train model....");
        startTime = System.currentTimeMillis();
        model.fit(data);
        endTime = System.currentTimeMillis();
        trainTime = (int) (endTime - startTime) / 60000;
    }

    private void evaluatePerformance(MultipleEpochsIterator iter){
        System.out.println("Evaluate model....");

        startTime = System.currentTimeMillis();
        Evaluation eval = model.evaluate(iter);
        endTime = System.currentTimeMillis();
        System.out.println(eval.stats(true));
        System.out.println("****************************************************");
        testTime = (int) (endTime - startTime) / 60000;

    }


}
