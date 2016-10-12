package imagenet.Utils;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 *
 */
public class ImageNetRecordReader extends BaseImageRecordReader {

    protected ImageNetLoader loader;
    protected Map<String, String> labelFileIdMap = new LinkedHashMap<>();
    protected DataModeEnum dataModeEnum = DataModeEnum.CLS_TRAIN; // use to load label ids for validation data set

    public ImageNetRecordReader(DataModeEnum dataModeEnum, int batchSize, int numExamples, int numLabels, int maxExamplesPerLabel, int height, int width, int channels, PathLabelGenerator labelGenerator, double splitTrainTest, Random rng) {
        super(height, width, channels, null, null);
        this.loader = new ImageNetLoader(batchSize, numExamples, numLabels, maxExamplesPerLabel, dataModeEnum, labelGenerator, splitTrainTest, rng);
        this.labelGenerator = labelGenerator; // Do not append when initializing split but use locally
        this.dataModeEnum = dataModeEnum;
        this.labelSetup();
    }

    private Map<String, String> defineLabels(String path) throws IOException {
        Map<String, String> tmpMap = new LinkedHashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;

        while ((line = br.readLine()) != null) {
            String row[] = line.split(",");
            tmpMap.put(row[0], row[1]);
            labels.add(row[1]);
        }
        return tmpMap;
    }

    private void labelSetup() {
        // creates hashmap with WNID (synset id) as key and first descriptive word in list as the string name
        if (labelFileIdMap.isEmpty()) {
            try {
                labelFileIdMap = defineLabels(ImageNetLoader.BASE_DIR + ImageNetLoader.CLS_TRAIN_ID_TO_LABELS);
            } catch (IOException e) {
                e.printStackTrace();
            }
            labels = new ArrayList<>(labelFileIdMap.values());
        }
        // creates hasmap with filename as key and WNID(synset id) as value when using val files
        if ((dataModeEnum == DataModeEnum.CLS_VAL || dataModeEnum == DataModeEnum.DET_VAL) && fileNameMap.isEmpty()) {
            try {
                fileNameMap = defineLabels(ImageNetLoader.BASE_DIR + ImageNetLoader.CLS_VAL_ID_TO_LABELS);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public List<Writable> next() {
        if (iter != null) {
            List<Writable> ret = new ArrayList<>();
            File image = iter.next();

            if (image.isDirectory())
                return next();

            try {
                invokeListeners(image);
                ret = setUpRecord(imageLoader.asMatrix(image), image.getName());
            } catch (Exception e) {
                e.printStackTrace();
            }
            return ret;
        } else if (record != null) {
            hitImage = true;
            invokeListeners(record);
            return record;
        }
        throw new IllegalStateException("No more elements");
    }

    private List<Writable> setUpRecord(INDArray image, String filename) throws IOException {
        int labelId;
        List<Writable> ret = RecordConverter.toRecord(image);
        if (dataModeEnum != DataModeEnum.CLS_VAL || dataModeEnum != DataModeEnum.DET_VAL) {
            Writable WNID = labelGenerator.getLabelForPath(filename);
            labelId = labels.indexOf(labelFileIdMap.get(WNID.toString()));
        } else {
            String fileName = FilenameUtils.getName(filename); // currently expects file extension
            labelId = labels.indexOf(labelFileIdMap.get(fileNameMap.get(fileName)));
        }
        if (labelId >= 0)
            ret.add(new IntWritable(labelId));
        else
            throw new IllegalStateException("Illegal label " + labelId);
        return ret;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        labelSetup();
        return setUpRecord(imageLoader.asRowVector(dataInputStream), FilenameUtils.getName(uri.getPath()));
    }

    public ImageNetRecordReader getSplit(ImageTransform imageTransform, int splitPosition) throws Exception{
        InputSplit split = loader.getSplit(splitPosition);
        initialize(split, imageTransform);
        return this;
    }


}
