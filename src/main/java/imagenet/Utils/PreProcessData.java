package imagenet.Utils;


import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.input.PortableDataStream;

import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.data.FilesAsBytesFunction;
import org.datavec.spark.functions.data.RecordReaderBytesFunction;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * Prep data as sequence files to group small files into larger batches to enable
 * optimized processing with Spark and Hadoop.
 *
 */
public class PreProcessData {

    public static final String TEMP_DIR = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "ImageNetSeqFiles");
    protected static String SEQUENCE_DIR;
    protected JavaPairRDD<Text, BytesWritable> sequenceFile;
    protected boolean save = false;
    protected JavaSparkContext sc;

    public PreProcessData(boolean save){
        this(null, save);
    }

    public PreProcessData(JavaSparkContext sc, boolean save){
        this.save = save;
        this.sc = sc != null? sc : setupLocalSpark();
    }

    private JavaSparkContext setupLocalSpark(){
        // Spark configuration
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("DataSequence");
        sparkConf.setMaster("local[*]");
        return new JavaSparkContext(sparkConf);
    }

    private void saveFiles(String outputPath){
        SEQUENCE_DIR = outputPath != null? outputPath: FilenameUtils.concat(TEMP_DIR, "tmp");
        if(new File(outputPath).exists()){
            try {
                FileUtils.deleteDirectory(new File(outputPath));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        sequenceFile.saveAsNewAPIHadoopFile(SEQUENCE_DIR, Text.class, BytesWritable.class, SequenceFileOutputFormat.class);
        System.out.println("\n****************Sequence files saved at " + outputPath.toString() + " ********************");
    }

    public void setupSequnceFile(String inputPath, String outputPath){
        // pass in file path with *.[file type] (e.g. jpg) to limit type of files loaded or use main dir and ensure only image files in subDir

        JavaPairRDD<String, PortableDataStream> stream = sc.binaryFiles(inputPath);
        sequenceFile = stream.mapToPair(new FilesAsBytesFunction());
        sequenceFile.cache();
        System.out.println("\n************Sequence file loaded************************");
        // Save SequenceFiles:
        if(save) {
            saveFiles(outputPath);
            sequenceFile.unpersist();
        }
    }

    // Doesn't work...
//    public void setupSequnceFile(List<File> files, String outputPath){
//        JavaRDD<File> stream = sc.parallelize(files);
//        sequenceFile = stream.mapToPair(new PairFunction<File, Text, BytesWritable>() {
//            @Override
//            public Tuple2<Text, BytesWritable> call(File file) throws Exception {
//                byte[] bytes = Files.toByteArray(file);
//                return new Tuple2<>(new Text(file.toString()), new BytesWritable(bytes));
//            }
//        });
//        sequenceFile.cache();
//        System.out.println("\n************Sequence file loaded************************");
//        // Save SequenceFiles:
//        if(save) {
//            saveFiles(outputPath);
//        }
//    }

    // TODO remove - temp to test results - Rework
    public void checkFile(String inputPath, DataModeEnum dataModeEnum){
//        JavaPairRDD<Text, BytesWritable> data = sc.sequenceFile(inputPath, Text.class, BytesWritable.class);
//        RecordReaderBytesFunction recordReaderFunc = new RecordReaderBytesFunction(
//                new ImageNetRecordReader(40, 40, 3, null, null, 255, dataModeEnum));
//        JavaRDD<List<Writable>> rdd = data.map(recordReaderFunc);
//        JavaRDD<DataSet> ds = rdd.map(new DataVecDataSetFunction(-1, 1860, false));
    }

    public JavaPairRDD<Text, BytesWritable> getFile(){
        return sequenceFile;
    }

    public void unpersist(){
        sequenceFile.unpersist();
    }

}
