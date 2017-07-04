package imagenet.Utils;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang.NotImplementedException;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.PatternPathLabelGenerator;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Loader specific to this project.
 */

public class ImageNetLoader extends NativeImageLoader implements Serializable {

    public final static int NUM_CLS_TRAIN_IMAGES = 1281167;
    public final static int NUM_CLS_VAL_IMAGES = 50000;
    public final static int NUM_CLS_TEST_IMAGES = 100000;
    public final static int NUM_CLS_LABELS = 5; // 1000 main with 860 ancestors ��������������ʶ���labels 1861 by zl

    public final static int NUM_DET_TRAIN_IMAGES = 395918;
    public final static int NUM_DET_VAL_IMAGES = 20121;
    public final static int NUM_DET_TEST_IMAGES = 40152;

    public final static int WIDTH = 112;//by zl
    public final static int HEIGHT = 112;
 // channels refer to the color depth of the image, 1 for greyscale, 3 for RGB
    public final static int CHANNELS = 3;

    public final static String BASE_DIR = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
    public final static String LOCAL_TRAIN_DIR = "train";
    public final static String LOCAL_VAL_DIR = "test";
    public final static String CLS_TRAIN_ID_TO_LABELS = "cls-loc-labels.txt";
    public final static String CLS_VAL_ID_TO_LABELS = "cls-loc-val-map.txt";
    public String urlTrainFile = "image_train_urls.txt";
    public String urlValFile = "image_test_urls.txt";
    protected String labelFilePath;

    protected List<String> labels = new ArrayList<>();
    protected Map<String, String> labelIdMap = new LinkedHashMap<>();

    protected File fullTrainDir = new File(BASE_DIR, LOCAL_TRAIN_DIR);
    protected File fullTestDir = new File(BASE_DIR, LOCAL_VAL_DIR);
    protected File sampleURLTrainList = new File(BASE_DIR, urlTrainFile);
    protected File sampleURLTestList = new File(BASE_DIR, urlValFile);

    protected File fullDir;
    protected File urlList;
    protected InputSplit[] inputSplit;
    protected int batchSize;
    protected int numExamples;
    protected int numLabels;
    protected int maxExamples2Label;
    protected PathLabelGenerator labelGenerator;
    protected double splitTrainTest;
    protected Random rng;

    protected DataModeEnum dataModeEnum; // CLS_Train, CLS_VAL, CLS_TEST, DET_TRAIN, DET_VAL, DET_TEST
    protected final static String REGEX_PATTERN = Pattern.quote("_");
    public final static PathLabelGenerator LABEL_PATTERN = new PatternPathLabelGenerator(REGEX_PATTERN);

    public ImageNetLoader(int batchSize, int numExamples, int numLabels, int maxExamples2Label, DataModeEnum dataModeEnum, PathLabelGenerator labelGenerator, double splitTrainTest, Random rng) {
        this( batchSize, numExamples, numLabels, maxExamples2Label, labelGenerator, dataModeEnum, splitTrainTest, rng, null);
    }

    public ImageNetLoader(int batchSize, int numExamples, int numLabels, int maxExamples2Label, PathLabelGenerator labelGenerator, DataModeEnum dataModeEnum, double splitTrainTest, Random rng, File localDir) {
        this.batchSize = batchSize;
        this.numExamples = numExamples;
        this.numLabels = numLabels;
        this.maxExamples2Label = maxExamples2Label;
        this.labelGenerator = labelGenerator == null ? LABEL_PATTERN : labelGenerator;
        this.labelFilePath = (dataModeEnum == DataModeEnum.CLS_VAL || dataModeEnum == DataModeEnum.DET_VAL) ? CLS_VAL_ID_TO_LABELS : CLS_TRAIN_ID_TO_LABELS;
        this.splitTrainTest = Double.isNaN(splitTrainTest) ? 1 : splitTrainTest;
        this.rng = rng == null ? new Random(System.currentTimeMillis()) : rng;
        this.dataModeEnum = dataModeEnum;
        switch (dataModeEnum) {
            case CLS_TRAIN:
                this.fullDir = localDir == null ? fullTrainDir : localDir;
                this.urlList = sampleURLTrainList;
                break;
            case CLS_TEST:
                this.fullDir = localDir == null ? fullTestDir : localDir;
                this.urlList = sampleURLTestList;
                break;
            case CLS_VAL:
            case DET_TRAIN:
            case DET_VAL:
            case DET_TEST:
                throw new NotImplementedException("Detection has not been setup yet");
            default:
                break;
        }
        setupInputSplit();
    }

    @Override
    public INDArray asRowVector(File f) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asRowVector(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asMatrix(File f) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray asMatrix(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    public Map<String, String> generateMaps(String filesFilename, String url) {
        Map<String, String> imgNetData = new HashMap<>();
        imgNetData.put("filesFilename", filesFilename);
        imgNetData.put("filesURL", url);
        return imgNetData;
    }

    private void defineLabels(File labelFilePath) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(labelFilePath));
            String line;

            while ((line = br.readLine()) != null) {
                String row[] = line.split(",");
                labelIdMap.put(row[0], row[1]);
                labels.add(row[1]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setupInputSplit() {
        defineLabels(new File(BASE_DIR, labelFilePath));
        // Downloading a sample set of data if not available
        if (!fullDir.exists()) {
            fullDir.mkdir();

            CSVRecordReader reader = new CSVRecordReader(7, ",");
            log.info("Checking files in the dir {} one by one, then download or not ... ", FilenameUtils.getBaseName(fullDir.toString()));
            try {
                reader.initialize(new FileSplit(urlList));
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
            int count = 0;
            while (reader.hasNext()) {
                Collection<Writable> val = reader.next();
                //����Ƿ��,��Ϊ��������
                /**
                 * ���ش���
                 * -Dhttp.proxyHost=127.0.0.1 -Dhttp.proxyPort=56731 by zl
                 */
                String url = val.toArray()[1].toString();
                String localUrl = null;
                if(url.indexOf("localhost")==-1)
                	localUrl = "http://localhost/imageNet/"+url.substring(url.lastIndexOf("/")+1);
                else
                	localUrl = url;
                String fileName = val.toArray()[0] + "_" + count++ + ".jpg";
                try {
                    downloadAndUntar(generateMaps(fileName, localUrl), fullDir);
                    System.out.println(">>>>>>"+fileName+"<<<<<<");
                } catch (Exception e) {
                    log.error("fileName is {}, url: {} is cann`t download", fileName, url);
                    System.err.println(fileName);
//                    e.printStackTrace();
//                    throw e;
                }
            }
        }
        FileSplit fileSplit = new FileSplit(fullDir, ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, ALLOWED_FORMATS, labelGenerator, numExamples, numLabels, maxExamples2Label, maxExamples2Label, null);
        inputSplit = fileSplit.sample(pathFilter, numExamples * splitTrainTest, numExamples * (1 - splitTrainTest));
    }

    public List<String> getLabels() {
        return labels;
    }

    public InputSplit getSplit(int splitPosition) throws Exception{
         return inputSplit[splitPosition];
    }


}
