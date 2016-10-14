package imagenet;

import imagenet.Models.*;
import imagenet.Utils.DataModeEnum;
import imagenet.Utils.ImageNetLoader;
import imagenet.Utils.ImageNetRecordReader;
import org.apache.commons.io.FilenameUtils;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.NetSaverLoaderUtils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * ImageNet is a large scale visual recognition challenge run by Stanford and Princeton. The competition covers
 * standard object classification as well as identifying object location in the image.
 *
 * This file is the main class that is called when running the program. Pass in arguments to help
 * adjust how and where the program will run. Note ImageNet is typically structured with 224 x 224
 * pixel size images but this can be adjusted by change WIDTH & HEIGHT.
 *
 * References: ImageNet
 * Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
 * Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei.
 * (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
 *
 */
public class ImageNetMain {
    private static final Logger log = LoggerFactory.getLogger(ImageNetMain.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--useSpark",usage="Use Spark",aliases = "-sp")
    protected boolean useSpark = false;
    @Option(name="--modelType",usage="Type of model (AlexNet, VGGNetA, VGGNetB)",aliases = "-mT")
    protected String modelType = "AlexNet";
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    protected int batchSize = 80;
    @Option(name="--maxExamplesPerLabelTrain",usage="Max examples per label",aliases="-mBai")
    protected int maxExamplesPerLabelTrain = 20;
    @Option(name="--maxExamplesPerLabelTest",usage="Max examples per label",aliases="-mBes")
    protected int maxExamplesPerLabelTest = 20;
    @Option(name="--numBatches",usage="Number of batches",aliases="-nB")
    protected int numBatches = 1;
    @Option(name="--numExamples",usage="Number of examples",aliases="-nEx")
    protected int numExamples = 210; // batchSize * numBatches;
    @Option(name="--splitTrainTest",usage="Percent to split for train and test (how much goes to train)",aliases="-split")
    protected double splitTrainTest = 1.0; // Different files for train and test
    @Option(name="--numEpochs",usage="Number of epochs",aliases="-nE")
    protected int numEpochs = 20;
    @Option(name="--iterations",usage="Number of iterations",aliases="-i")
    protected int iterations = 1;
    @Option(name="--trainFolder",usage="Train folder",aliases="-taF")
    protected String trainFolder = "train";
    @Option(name="--testFolder",usage="Test folder",aliases="-teF")
    protected String testFolder = "test";
    @Option(name="--saveModel",usage="Save model",aliases="-sM")
    protected boolean saveModel = false;
    @Option(name="--saveParams",usage="Save parameters",aliases="-sP")
    protected boolean saveParams = false;
    @Option(name="--saveUpdater",usage="Save updater",aliases="-sU")
    protected boolean saveUpdater = false;

    @Option(name="--confName",usage="Model configuration file name",aliases="-conf")
    protected String confName;
    @Option(name="--paramName",usage="Parameter file name",aliases="-param")
    protected String paramName;

    protected long startTime;
    protected long endTime;
    protected int trainTime;
    protected int testTime;

    protected static final int height = ImageNetLoader.HEIGHT;
    protected static final int width = ImageNetLoader.WIDTH;
    protected static final int channels = ImageNetLoader.CHANNELS;
    protected static final int numLabels = ImageNetLoader.NUM_CLS_LABELS;
    protected int seed = 42;
    protected Random rng = new Random(seed);
    protected int listenerFreq = 1;
    protected int numCores = 5;

    // Paths for data
    protected String basePath = ImageNetLoader.BASE_DIR;
    protected String trainPath = FilenameUtils.concat(basePath, trainFolder);
    protected String testPath = FilenameUtils.concat(basePath, testFolder);

//        String trainPath = FilenameUtils.concat(new ClassPathResource("train").getFile().getAbsolutePath(), "*");
//        String testPath = FilenameUtils.concat(new ClassPathResource("test").getFile().getAbsolutePath(), "*");
//    protected String trainPath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/train/*");
//    protected String testPath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/" + testFolder + "/*");

    protected String outputPath = NetSaverLoaderUtils.defineOutputDir(modelType.toString());
    protected String updaterPath = NetSaverLoaderUtils.defineOutputDir(modelType.toString() + "Updater");
    protected Map<String, String> paramPaths = new HashMap<>();
    protected String[] layerNames; // Names of layers to store parameters
    protected String rootParamPath;

    protected String sparkMasterUrl = "local[" + numCores + "]";

    protected MultiLayerNetwork model = null;
    protected ComputationGraph modelCG = null;

    public void run(String[] args) throws Exception {
        Nd4j.dtype = DataBuffer.Type.FLOAT;
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        if(useSpark) {
            System.out.println("Spark functions our outdated and need overhaul to include major changes in 2016.");
        } else {
            // Build
            buildModel();
            setListeners();

            // Train
            ImageTransform flipTransform = new FlipImageTransform(new Random(42));
            ImageTransform warpTransform = new WarpImageTransform(new Random(42), 42);
            List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});
            ImageNetRecordReader reader = new ImageNetRecordReader(DataModeEnum.CLS_TRAIN, batchSize, numExamples, numLabels, maxExamplesPerLabelTrain, height, width, channels, ImageNetLoader.LABEL_PATTERN, splitTrainTest, rng);
            DataSetIterator iter;
            DataNormalization scaler = new ImagePreProcessingScaler();

            MultipleEpochsIterator trainIter;

            for(ImageTransform transform: transforms) {
                log.info("Training with " + (transform == null? "no": transform.toString()) + " transform");
                iter = new RecordReaderDataSetIterator(reader.getSplit(transform, 0), batchSize, 1, numLabels);
                scaler.fit(iter);
                iter.setPreProcessor(scaler);
                trainIter = new MultipleEpochsIterator(numEpochs, iter);
                trainTime = trainModel(trainIter);
            }

            // Evaluation
            reader = new ImageNetRecordReader(DataModeEnum.CLS_TEST, batchSize, numExamples, numLabels, maxExamplesPerLabelTest, height, width, channels, ImageNetLoader.LABEL_PATTERN, splitTrainTest, rng);
            iter = new RecordReaderDataSetIterator(reader.getSplit(null, 0), batchSize, 1, numLabels);
            scaler.fit(iter);
            iter.setPreProcessor(scaler);
            MultipleEpochsIterator testIter = new MultipleEpochsIterator(1, iter);
            testTime = evaluatePerformance(testIter);

            // Save
            saveAndPrintResults();

        }

        System.out.println("****************Example finished********************");
    }

    protected void buildModel() {
        System.out.println("Build model....");
        if (confName != null && paramName != null) {
            String confPath = FilenameUtils.concat(outputPath, confName + "conf.yaml");
            String paramPath = FilenameUtils.concat(outputPath, paramName + "param.bin");
            model = NetSaverLoaderUtils.loadNetworkAndParameters(confPath, paramPath);
        } else {
            switch (modelType) {
                case "LeNet":
                    model = new LeNet(height, width, channels, numLabels, seed, iterations).init();
                    break;
                case "AlexNet":
                    model = new AlexNet(height, width, channels, numLabels, seed, iterations).init();
                    break;
                case "VGGNetA":
                    model = new VGGNetA(height, width, channels, numLabels, seed, iterations).init();
                    break;
                case "VGGNetD":
                    model = new VGGNetD(height, width, channels, numLabels, seed, iterations, rootParamPath).init();
                    break;
                case "GoogleLeNet":
                    modelCG = new GoogleLeNet(height, width, channels, numLabels, seed, iterations).init();
                default:
                    break;
            }
        }
    }

    protected void setListeners(){
        // Listeners
        IterationListener paramListener = ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(1).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();

        if(model != null) model.setListeners(new ScoreIterationListener(listenerFreq)); // not needed for spark?
        else modelCG.setListeners(new ScoreIterationListener(listenerFreq));
//        model.setListeners(new HistogramIterationListener(1));
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), paramListener));

    }


    private int trainModel(MultipleEpochsIterator data){
        System.out.println("Train model....");
        startTime = System.currentTimeMillis();
        if(model != null) model.fit(data);
        else modelCG.fit(data);
        endTime = System.currentTimeMillis();
        return (int) (endTime - startTime);
    }

    private int evaluatePerformance(MultipleEpochsIterator iter){
        System.out.println("Evaluate model....");
        startTime = System.currentTimeMillis();
        Evaluation eval = (model != null)? model.evaluate(iter): modelCG.evaluate(iter);
        endTime = System.currentTimeMillis();
        System.out.println(eval.stats(true));
        return (int) (endTime - startTime);

    }

    protected void saveAndPrintResults(){
        System.out.println("****************************************************");
        printTime("train", trainTime);
        printTime("test", testTime);
        System.out.println("Total evaluation runtime: " + testTime + " minutes");
        System.out.println("****************************************************");
        if (saveModel) NetSaverLoaderUtils.saveNetworkAndParameters(model, outputPath.toString());
        if (saveParams) NetSaverLoaderUtils.saveParameters(model, model.getLayerNames().toArray(new String[]{}), paramPaths);
        if (saveUpdater) NetSaverLoaderUtils.saveUpdators(model, updaterPath);


    }

    public static void printTime(String name, long ms){
        log.info(name + " time: {} min, {} sec | {} milliseconds",
                TimeUnit.MILLISECONDS.toMinutes(ms),
                TimeUnit.MILLISECONDS.toSeconds(ms) -
                        TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(ms)),
                ms);
    }

    public static void main(String[] args) throws Exception {
        new ImageNetMain().run(args);
    }


}
