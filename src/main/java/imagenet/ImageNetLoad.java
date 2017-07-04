package imagenet;

import java.io.File;
import java.text.DecimalFormat;

import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ImageNetLoad {

	MultiLayerNetwork model;

	public void init() throws Exception {
		File locationModelFile = new File(
				"/tmp/trained_mnist_model.zip");
		model = ModelSerializer.restoreMultiLayerNetwork(locationModelFile);
	}

	public void fit() throws Exception {
		// ImageRecordReader imageRecord = new ImageRecordReader(21, 21, 3);
		// imageRecord.initialize(new FileSplit(new File("")));
		ImageLoader imageLoader = new ImageLoader(112, 112, 3);
		INDArray ind1 = model.output(imageLoader.asRowVector(new File(
				"/tmp/canny_1_40.JPG")));
		System.out.println(ind1);
//		INDArray ind2 = model.output(imageLoader.asRowVector(new File(
//				"D:\\test\\car\\realloc\\jpg\\canny\\canny_3_133.JPG")));
//		System.out.println(ind2);
//		INDArray ind3 = model.output(imageLoader.asRowVector(new File(
//				"D:\\test\\car\\realloc\\jpg\\canny\\canny_3_265.JPG")));
//		System.out.println(ind3);
//		INDArray ind4 = model.output(imageLoader.asRowVector(new File(
//				"D:\\test\\car\\realloc\\jpg\\canny\\canny_5_6.JPG")));
//		System.out.println(ind4);
//		INDArray ind5 = model.output(imageLoader.asRowVector(new File(
//				"D:\\test\\car\\realloc\\jpg\\canny\\canny_5_161.JPG")));
//		System.out.println(ind5);
//		INDArray ind6 = model.output(imageLoader.asRowVector(new File(
//				"D:\\test\\car\\realloc\\jpg\\canny\\canny_9_94.JPG")));
//		System.out.println(ind6);
//		if (ind5.isVector()) {
//			DecimalFormat decimalFormat = new DecimalFormat("0.000");
//			for (int i = 0; i < ind5.length(); i++) {
//				System.out.println(decimalFormat.format(ind5.getDouble(i)));
//			}
//		}
	}

	/**
	 * 加载训练好的模型和参数
	 * 
	 * @auth zhanglei
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			ImageNetLoad inl = new ImageNetLoad();
			inl.init();
			inl.fit();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
