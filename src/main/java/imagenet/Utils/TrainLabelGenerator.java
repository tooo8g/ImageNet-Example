package imagenet.Utils;

import org.apache.commons.io.FilenameUtils;
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
public class TrainLabelGenerator implements PathLabelGenerator {
    private Map<String,Integer> labelIdxs;

    public TrainLabelGenerator(List<String> labels) throws IOException {
        labelIdxs = new HashMap<>();
        int i=0;
        for(String s : labels){
            labelIdxs.put(s, i++);
        }
    }

    @Override
    public Writable getLabelForPath(String path) {
        String dirName = FilenameUtils.getBaseName(new File(path).getParentFile().getParent());
        return new Text(dirName);
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(uri.toString());
    }

}
