package imagenet;

import imagenet.Utils.DataModeEnum;
import imagenet.Utils.ImageNetDataSetIterator;
import imagenet.Utils.ImageNetRecordReader;
import imagenet.Utils.PreProcessData;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.io.BytesWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.writable.Writable;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.datavec.spark.functions.data.RecordReaderBytesFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.apache.hadoop.io.Text;

import java.util.*;


/**
 * Spark configuration to run ImageNet. The version argument from CNNImageNetMain sets whether it will run
 * SparkStandalone on just a local machine or SparkCluster on a cluster with master and workers.
 */
public class SparkSupportFunctions {
    private static final Logger log = LoggerFactory.getLogger(SparkSupportFunctions.class);


//    public void initialize() throws Exception{
//        // Spark context
//        JavaSparkContext sc = setupSpark();
//
//        // Load data and train
//        String seqOutputPath = FilenameUtils.concat(PreProcessData.TEMP_DIR, "tmp");
//        SparkDl4jMultiLayer sparkNetwork = null;
//        JavaRDD<DataSet> trainData = null;
//        ImageTransform flipTransform = new FlipImageTransform(rng);
//        ImageTransform warpTransform = new WarpImageTransform(rng, seed);
//        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});
//
//        //Setup parameter averaging
//        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
//                .workerPrefetchNumBatches(0)
//                .saveUpdater(true)
//                .averagingFrequency(5)
//                .batchSizePerWorker(batchSize)
//                .build();
//
//        // Build
//        buildModel();
//        setListeners();
//
//        // Train
//        for(ImageTransform transform: transforms) {
//            trainData = loadData(sc, trainPath, seqOutputPath, numTrainExamples, false, transform, DataModeEnum.CLS_TRAIN);
//            sparkNetwork = trainModel(new SparkDl4jMultiLayer(sc, model, trainMaster), trainData);
//        }
//        // Eval
//        JavaRDD<DataSet> testData = loadData(sc, testPath, seqOutputPath, numTestExamples, false, null, DataModeEnum.CLS_TEST);
//        evaluatePerformance(sparkNetwork, testData);
//
//        // Save
//        saveAndPrintResults();
//
//        // Close
//        cleanUp(trainData);
//        cleanUp(testData);
//    }
//
//    private JavaSparkContext setupSpark(){
//        SparkConf conf = new SparkConf()
//                .setMaster(sparkMasterUrl);
//        conf.setAppName("ImageNet Local");
////        conf.set("spak.executor.memory", "4g");
////        conf.set("spak.driver.memory", "4g");
////        conf.set("spark.driver.maxResultSize", "1g");
////        conf.set(SparkDl4jMultiLayer.ACCUM_GRADIENT, String.valueOf(true));
//        return new JavaSparkContext(conf);
//    }
//
//
//    private JavaRDD<DataSet> loadData(JavaSparkContext sc, String inputPath, String seqOutputPath, int numExamples, boolean save, ImageTransform transform, DataModeEnum dataModeEnum) {
//        System.out.println("Load data...");
//
//        JavaPairRDD<Text, BytesWritable> filesAsBytes = null;
//        JavaRDD<DataSet> data;
//
//        if(inputPath==null && seqOutputPath != null){
//            filesAsBytes = sc.sequenceFile(seqOutputPath, Text.class, BytesWritable.class);
////        } else if(version == "SparkStandAlone"){
////            PreProcessData pData = new PreProcessData(sc, save);
////            pData.setupSequnceFile(inputPath, seqOutputPath);
////            filesAsBytes = pData.getFile();
//        } else {
//            ImageNetDataSetIterator img = new ImageNetDataSetIterator(batchSize, numExamples,
//            new int[] {HEIGHT, WIDTH, CHANNELS}, numLabels, maxExamples2Label, dataModeEnum, splitTrainTest, transform, normalizeValue, rng);
//            List<DataSet> dataList = new ArrayList<>();
//            while(img.hasNext()){
//                dataList.add(img.next());
//            }
//            return sc.parallelize(dataList);
////            throw new IllegalArgumentException("Data can not be loaded running on a cluster without an outputPath.");
//        }
//
//        RecordReaderBytesFunction recordReaderFunc = new RecordReaderBytesFunction(
//                    new ImageNetRecordReader(HEIGHT, WIDTH, CHANNELS, null, transform, normalizeValue, dataModeEnum));
//
//        JavaRDD<List<Writable>> rdd = filesAsBytes.map(recordReaderFunc);
//        // Load all files in path
//        if(numExamples==-1)
//            data = rdd.map(new DataVecDataSetFunction(-1, numLabels, false));
//        else {
//            // Limit number examples loaded
//            JavaRDD<DataSet> dataRdd = rdd.map(new DataVecDataSetFunction(-1, numLabels, false));
//            List<DataSet> listData = dataRdd.take(numExamples); // should have features and labels (1*1860) filled out
//            data = sc.parallelize(listData);
//        }
//
//        data.cache();
//        filesAsBytes.unpersist();
//        return data;
//    }
//
//    private SparkDl4jMultiLayer trainModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data){
//        System.out.println("Train model...");
//        startTime = System.currentTimeMillis();
//        model.fit(data);
//        endTime = System.currentTimeMillis();
//        trainTime = (int) (endTime - startTime) / 60000;
//        return model;
//
//    }
//
//    private void evaluatePerformance(SparkDl4jMultiLayer model, JavaRDD<DataSet> testData) {
//        System.out.println("Eval model...");
//        startTime = System.currentTimeMillis();
//        Evaluation evalActual = model.evaluate(testData);
//        System.out.println(evalActual.stats());
//        endTime = System.currentTimeMillis();
//        testTime = (int) (endTime - startTime) / 60000;
//    }
//

//    public void cleanUp(JavaRDD<DataSet> data) {
//        data.unpersist();
//    }

}
