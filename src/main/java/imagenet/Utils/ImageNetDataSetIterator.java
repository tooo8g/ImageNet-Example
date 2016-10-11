package imagenet.Utils;


import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import java.io.File;
import java.util.Random;


/**
 * DL4J DataSetIterator specific to this project.
 */

public class ImageNetDataSetIterator extends RecordReaderDataSetIterator {

    /** Loads images with given  batchSize, numExamples returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples) {
        this(batchSize, numExamples, new int[] {ImageNetLoader.HEIGHT, ImageNetLoader.WIDTH, ImageNetLoader.CHANNELS }, ImageNetLoader.NUM_CLS_LABELS, 0, ImageNetLoader.LABEL_PATTERN,  DataModeEnum.CLS_TRAIN, 1, null, 255, new Random(System.currentTimeMillis()), null);
    }

    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, dataMode returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, DataModeEnum dataModeEnum) {
        this(batchSize, numExamples, imgDim, numLabels, 0, ImageNetLoader.LABEL_PATTERN, dataModeEnum, 1, null, 0, new Random(System.currentTimeMillis()), null);
    }

    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, dataMode, train, splitTrainTest, imageTransform, normalizeValue, rng returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, DataModeEnum dataModeEnum, double splitTrainTest, ImageTransform imageTransform, int normalizeValue, Random rng) {
        this(batchSize, numExamples, imgDim, numLabels, 0, ImageNetLoader.LABEL_PATTERN, dataModeEnum, splitTrainTest, imageTransform, normalizeValue, rng, null);
    }

    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, dataMode, train, splitTrainTest, imageTransform, normalizeValue, rng returned by the generator. */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, int maxExample2Label, DataModeEnum dataModeEnum, double splitTrainTest, ImageTransform imageTransform, int normalizeValue, Random rng) {
        this(batchSize, numExamples, imgDim, numLabels, maxExample2Label, ImageNetLoader.LABEL_PATTERN, dataModeEnum, splitTrainTest, imageTransform, normalizeValue, rng, null);
    }

    /**
     * Create ImageNet data specific iterator
     * @param batchSize the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param imgDim an array of width, height and channels
     * @param numLabels the overall number of examples
     * @param maxExamplesPerLabel max num examples to a label
     * @param dataModeEnum which type of data to load CLS_TRAIN, CLS_VAL, DET_TRAIN, DET_VAL
     * @param labelGenerator path label generator to use
     * @param splitTrainTest the percentage to split data for train and remainder goes to test
     * @param normalizeValue value to divide pixels by to normalize
     * @param localDir File path to an explicit directory
     * @param rng random number to lock in batch shuffling

     * */
    public ImageNetDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numLabels, int maxExamplesPerLabel, PathLabelGenerator labelGenerator, DataModeEnum dataModeEnum, double splitTrainTest, ImageTransform imageTransform, int normalizeValue, Random rng, File localDir) {
        super(new ImageNetRecordReader(dataModeEnum, batchSize, numExamples, numLabels, maxExamplesPerLabel, imgDim[0], imgDim[1], imgDim[2], labelGenerator, splitTrainTest, rng), batchSize, 1, numLabels);
    }

}
