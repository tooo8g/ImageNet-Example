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
        boolean gradientCheck = false;

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

        // Gradient Check
        if (gradientCheck) gradientCheck(trainIter, model);

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

    private void gradientCheck(DataSetIterator dataIter, MultiLayerNetwork model){
        DataSet imgNet;
        System.out.println("Gradient Check....");

        imgNet = dataIter.next();
        String name = new Object() {
        }.getClass().getEnclosingMethod().getName();

        model.setInput(imgNet.getFeatures());
        model.setLabels(imgNet.getLabels());
        model.computeGradientAndScore();
        double scoreBefore = model.score();
        for (int j = 0; j < 1; j++)
            model.fit(imgNet);
        model.computeGradientAndScore();
        double scoreAfter = model.score();
//            String msg = name + " - score did not (sufficiently) decrease during learning (before=" + scoreBefore + ", scoreAfter=" + scoreAfter + ")";
//            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
        for (int j = 0; j < model.getnLayers(); j++)
            System.out.println("Layer " + j + " # params: " + model.getLayer(j).numParams());

        double default_eps = 1e-6;
        double default_max_rel_error = 0.25;
        double default_min_rel_error = 1e-14;
        boolean print_results = true;
        boolean return_on_first_failure = false;
        boolean useUpdater = true;

        boolean gradOK = GradientCheckUtil.checkGradients(model, default_eps, default_max_rel_error, default_min_rel_error,
                print_results, return_on_first_failure, imgNet.getFeatures(), imgNet.getLabels());

//        assertTrue(gradOK);

    }



}
