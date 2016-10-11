package imagenet.Utils;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 */
public class ValidationLabelGenerator implements PathLabelGenerator {
    private Map<String,Integer> labelIdxs;
    private Map<String,String> filenameToIndex;

    private ValidationLabelGenerator(List<String> labels, String annotationsFile) throws IOException {
        labelIdxs = new HashMap<>();
        int i=0;
        for(String s : labels){
            labelIdxs.put(s, i++);
        }
        this.filenameToIndex = loadValidationSetLabels(annotationsFile);
    }

    @Override
    public Writable getLabelForPath(String path) {
        File f = new File(path);
        String filename = f.getName();
        return new Text(filenameToIndex.get(filename));
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(uri.toString());
    }


    private static Map<String,String> loadValidationSetLabels(String path) throws IOException {
        Map<String,String> validation = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(path));
        for(String s : lines){
            String[] split = s.split("\t");
            validation.put(split[0],split[1]);
        }
        return validation;
    }
}
