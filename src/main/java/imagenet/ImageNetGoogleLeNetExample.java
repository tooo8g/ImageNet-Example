package imagenet;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import imagenet.Models.GoogleLeNet;
import imagenet.Utils.DataModeEnum;
import imagenet.Utils.ImageNetDataSetIterator;
import imagenet.Utils.ImageNetLoader;

public class ImageNetGoogleLeNetExample {
	
    protected static final int HEIGHT = ImageNetLoader.HEIGHT;
    protected static final int WIDTH = ImageNetLoader.WIDTH;
    protected static final int CHANNELS = ImageNetLoader.CHANNELS;
    protected static final int numLabels = ImageNetLoader.NUM_CLS_LABELS;
    protected static final int SEED = 42;
	protected static final int ITERATIONS = 1;
    protected static int listenerFreq = 1;
    protected static int batchSize = 40;
    protected static int testBatchSize = batchSize;
    protected static int numBatches = 1;
    protected static int numTestBatches = numBatches;
    protected static int numTrainExamples = batchSize * numBatches;
    protected static int numTestExamples = testBatchSize * numTestBatches;
    protected static int numEpochs = 5;
    protected static int maxExamples2Label = 10;
    protected static int asynQues = 5;
    protected static int normalizeValue = 255;
    protected static double splitTrainTest = 0.8;
    protected static int seed = 42;
    protected static Random rng = new Random(seed);

    private static final Logger log = LoggerFactory.getLogger(ImageNetGoogleLeNetExample.class);
    
    public static void main(String[] args) throws Exception {
    	ComputationGraph model = new GoogleLeNet(HEIGHT, WIDTH, CHANNELS, numLabels, SEED, ITERATIONS).init();

        // Listeners
    	/*
        ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(1).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();
		*/
    	
        model.setListeners(new ScoreIterationListener(listenerFreq)); // not needed for spark?
        
        // Train
        MultipleEpochsIterator trainIter = null;
        ImageTransform flipTransform = new FlipImageTransform(new Random(42));
        ImageTransform warpTransform = new WarpImageTransform(new Random(42), 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});
        
        long startTime;
        long endTime;
        long trainTime;
        
        for(ImageTransform transform: transforms) {
            log.info("Training with " + (transform == null? "no": transform.toString()) + " transform");
            System.out.println("Load data....");
            trainIter = new MultipleEpochsIterator(numEpochs,
                    new ImageNetDataSetIterator(batchSize, numTrainExamples,
                            new int[] {HEIGHT, WIDTH, CHANNELS}, numLabels, maxExamples2Label, DataModeEnum.CLS_TRAIN, splitTrainTest, transform, normalizeValue, rng), asynQues);
            
            System.out.println("Train model....");
            startTime = System.currentTimeMillis();
            model.fit(trainIter);
            endTime = System.currentTimeMillis();
            trainTime = (int) (endTime - startTime) / 1000;
            System.out.println("training time " + trainTime);
        }

        // Evaluation
        numEpochs = 1;
        MultipleEpochsIterator testIter = new MultipleEpochsIterator(numEpochs,
                new ImageNetDataSetIterator(batchSize, numTestExamples,
                        new int[] {HEIGHT, WIDTH, CHANNELS}, numLabels, maxExamples2Label, DataModeEnum.CLS_TEST, splitTrainTest, null, normalizeValue, rng), asynQues);
        
        System.out.println("Evaluate model....");

        startTime = System.currentTimeMillis();
        if(model.getLayers() == null || !(model.getLayers()[model.getLayers().length - 1] instanceof IOutputLayer)){
            throw new IllegalStateException("Cannot evaluate network with no output layer");
        }

        List<String> labelsList = testIter.getLabels();

        Evaluation e = (labelsList == null)? new Evaluation(): new Evaluation(labelsList);
        while(testIter.hasNext()){
            DataSet next = testIter.next();

            if (next.getFeatureMatrix() == null || next.getLabels() == null)
                break;

            INDArray features = next.getFeatures();
            INDArray labels = next.getLabels();

            INDArray[] out;
            out = model.output(false, features);
            if(labels.rank() == 3 ) e.evalTimeSeries(labels,out[0]);
            else e.eval(labels,out[0]);
        }

        Evaluation eval =  e;
        
        endTime = System.currentTimeMillis();
        System.out.println(eval.stats(true));
        System.out.println("****************************************************");
        long testTime = (int) (endTime - startTime) / 1000;
        System.out.println("testing time " + testTime);
        
        
    }
    
}
