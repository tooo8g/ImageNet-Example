package imagenet.Models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * GoogleLeNet
 *  Reference: http://arxiv.org/pdf/1409.4842v1.pdf
 *
 * Warning this has not been run yet.
 * There are a couple known issues with CompGraph regarding combining different layer types into one and
 * combining different shapes of input even if the layer types are the same at least for CNN.
 */

public class GoogleLeNet {

    private int height;
    private int width;
    private int channels = 3;
    private int outputNum = 1000;
    private long seed = 123;
    private int iterations = 90;

    public GoogleLeNet(int height, int width, int channels, int outputNum, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    private ConvolutionLayer conv1x1(int in, int out, double bias) {
    	    return new ConvolutionLayer.Builder(new int[] {1,1}, new int[] {1,1}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer c3x3reduce(int in, int out, double bias) {
    	return conv1x1(in, out, bias);
    }

    private ConvolutionLayer c5x5reduce(int in, int out, double bias) {
    	return conv1x1(in, out, bias);
    }

    private ConvolutionLayer conv3x3(int in, int out, double bias) {
    	return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(int in, int out, double bias) {
    	return new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {2,2}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv7x7(int in, int out, double bias) {
    	return new ConvolutionLayer.Builder(new int[]{7,7}, new int[]{2,2}, new int[]{3,3}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer avgPool7x7(int stride) {
    	return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{7,7}, new int[]{1,1}).build();
    }

    private SubsamplingLayer maxPool3x3(int stride) {
    	return new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{stride,stride}, new int[]{1,1}).build();
    }

    private DenseLayer fullyConnected(int in, int out, double dropOut) {
    	return new DenseLayer.Builder().nIn(in).nOut(out).dropOut(dropOut).build();
    }
    
    
    public ComputationGraph init() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        	    .seed(seed)
        	    .iterations(iterations)
        	    .activation("relu")
        	    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        	    .learningRate(1e-2)
        	    .biasLearningRate(2 * 1e-2)
        	    .learningRateDecayPolicy(LearningRatePolicy.Step)
        	    .lrPolicyDecayRate(0.96)
        	    .lrPolicySteps(320000)
        	    .updater(Updater.NESTEROVS)
        	    .momentum(0.9)
        	    .weightInit(WeightInit.XAVIER)
        	    .regularization(true)
        	    .l2(2e-4)        	    
        	    .graphBuilder()
        	    .addInputs("input")
        	    //.setInputTypes(InputType.convolutional(height,width,channels))
        	    .addLayer("cnn1", conv7x7(this.channels, 64, 0.2), "input")
        	    .addLayer("max1", maxPool3x3(2), "cnn1")
        	    .addLayer("lrn1", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "max1")
        	    .addLayer("cnn2", conv1x1(64, 64, 0.2), "lrn1")
        	    .addLayer("cnn3", conv3x3(64, 192, 0.2), "cnn2")
        	    .addLayer("lrn2", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "cnn3")
        	    .addLayer("max2", maxPool3x3(2), "lrn2")
        	    // Inception 3a
        	    .addLayer("cnn4", conv1x1(192, 64, 0.2), "max2")
        	    .addLayer("cnn5", c3x3reduce(192, 96, 0.2), "max2")
        	    .addLayer("cnn6", c5x5reduce(192, 16, 0.2), "max2")
        	    .addLayer("max3", maxPool3x3(1), "max2")
        	    .addLayer("cnn7", conv3x3(96, 128, 0.2), "cnn5")
        	    .addLayer("cnn8", conv5x5(16, 32, 0.2), "cnn6")
        	    .addLayer("cnn9", conv1x1(192, 32, 0.2), "max3")
        	    .addVertex("depthconcat1", new MergeVertex(), "cnn4", "cnn7", "cnn8", "cnn9")
        	    // Inception 3b
        	    .addLayer("cnn10", conv1x1(256, 128, 0.2), "depthconcat1")
        	    .addLayer("cnn11", c3x3reduce(256, 128, 0.2), "depthconcat1")
        	    .addLayer("cnn12", c5x5reduce(256, 32, 0.2), "depthconcat1")
        	    .addLayer("max4", maxPool3x3(1), "depthconcat1")
        	    .addLayer("cnn13", conv3x3(128, 192, 0.2), "cnn11")
        	    .addLayer("cnn14", conv5x5(32, 96, 0.2), "cnn12")
        	    .addLayer("cnn15", conv1x1(256, 64, 0.2), "max4")
        	    .addVertex("depthconcat2", new MergeVertex(), "cnn10", "cnn13", "cnn14", "cnn15")
        	    .addLayer("max5", maxPool3x3(2), "depthconcat2") // output: 28x28x192
        	    // Inception 4a
        	    .addLayer("cnn16", conv1x1(480, 192, 0.2), "max5")
        	    .addLayer("cnn17", c3x3reduce(480, 96, 0.2), "max5")
        	    .addLayer("cnn18", c5x5reduce(480, 16, 0.2), "max5")
        	    .addLayer("max6", maxPool3x3(1), "max5")
        	    .addLayer("cnn19", conv3x3(96, 208, 0.2), "cnn17")
        	    .addLayer("cnn20", conv5x5(16, 48, 0.2), "cnn18")
        	    .addLayer("cnn21", conv1x1(480, 64, 0.2), "max6")
        	    .addVertex("depthconcat3", new MergeVertex(), "cnn16", "cnn19", "cnn20", "cnn21") // output: 14x14x512
        	    // Inception 4b
        	    .addLayer("cnn22", conv1x1(512, 160, 0.2), "depthconcat3")
        	    .addLayer("cnn23", c3x3reduce(512, 112, 0.2), "depthconcat3")
        	    .addLayer("cnn24", c5x5reduce(512, 24, 0.2), "depthconcat3")
        	    .addLayer("max7", maxPool3x3(1), "depthconcat3")
        	    .addLayer("cnn25", conv3x3(112, 224, 0.2), "cnn23")
        	    .addLayer("cnn26", conv5x5(24, 64, 0.2), "cnn24")
        	    .addLayer("cnn27", conv1x1(512, 64, 0.2), "max7")
        	    .addVertex("depthconcat4", new MergeVertex(), "cnn22", "cnn25", "cnn26", "cnn27") // output: 14x14x512
        	    // Inception 4c
        	    .addLayer("cnn28", conv1x1(512, 128, 0.2), "depthconcat4")
        	    .addLayer("cnn29", c3x3reduce(512, 128, 0.2), "depthconcat4")
        	    .addLayer("cnn30", c5x5reduce(512, 24, 0.2), "depthconcat4")
        	    .addLayer("max8", maxPool3x3(1), "depthconcat4")
        	    .addLayer("cnn31", conv3x3(128, 256, 0.2), "cnn29")
        	    .addLayer("cnn32", conv5x5(24, 64, 0.2), "cnn30")
        	    .addLayer("cnn33", conv1x1(512, 64, 0.2), "max8")
        	    .addVertex("depthconcat5", new MergeVertex(), "cnn28", "cnn31", "cnn32", "cnn33") // output: 14x14x512
        	    // Inception 4d
        	    .addLayer("cnn34", conv1x1(512, 112, 0.2), "depthconcat5")
        	    .addLayer("cnn35", c3x3reduce(512, 144, 0.2), "depthconcat5")
        	    .addLayer("cnn36", c5x5reduce(512, 32, 0.2), "depthconcat5")
        	    .addLayer("max9", maxPool3x3(1), "depthconcat5")
        	    .addLayer("cnn37", conv3x3(144, 288, 0.2), "cnn35")
        	    .addLayer("cnn38", conv5x5(32, 64, 0.2), "cnn36")
        	    .addLayer("cnn39", conv1x1(512, 64, 0.2), "max9")
        	    .addVertex("depthconcat6", new MergeVertex(), "cnn34", "cnn37", "cnn38", "cnn39") // output: 14x14x528
        	    // Inception 4e
        	    .addLayer("cnn40", conv1x1(528, 256, 0.2), "depthconcat6")
        	    .addLayer("cnn41", c3x3reduce(528, 160, 0.2), "depthconcat6")
        	    .addLayer("cnn42", c5x5reduce(528, 32, 0.2), "depthconcat6")
        	    .addLayer("max10", maxPool3x3(1), "depthconcat6")
        	    .addLayer("cnn43", conv3x3(128, 320, 0.2), "cnn41")
        	    .addLayer("cnn44", conv5x5(24, 128, 0.2), "cnn42")
        	    .addLayer("cnn45", conv1x1(528, 128, 0.2), "max10")
        	    .addVertex("depthconcat7", new MergeVertex(), "cnn40", "cnn43", "cnn44", "cnn45") // output: 14x14x832
        	    .addLayer("max11", maxPool3x3(2), "depthconcat7")  // output: 7x7x832
        	    // Inception 5a
        	    .addLayer("cnn46", conv1x1(832, 256, 0.2), "max11")
        	    .addLayer("cnn47", c3x3reduce(832, 160, 0.2), "max11")
        	    .addLayer("cnn48", c5x5reduce(832, 32, 0.2), "max11")
        	    .addLayer("max12", maxPool3x3(1), "max11")
        	    .addLayer("cnn49", conv3x3(160, 320, 0.2), "cnn47")
        	    .addLayer("cnn50", conv5x5(32, 128, 0.2), "cnn48")
        	    .addLayer("cnn51", conv1x1(832, 128, 0.2), "max12")
        	    .addVertex("depthconcat8", new MergeVertex(), "cnn46", "cnn49", "cnn50", "cnn51") // output: 7x7x832
        	    // Inception 5b
        	    .addLayer("cnn52", conv1x1(832, 384, 0.2), "depthconcat8")
        	    .addLayer("cnn53", c3x3reduce(832, 192, 0.2), "depthconcat8")
        	    .addLayer("cnn54", c5x5reduce(832, 48, 0.2), "depthconcat8")
        	    .addLayer("max13", maxPool3x3(1), "depthconcat8")
        	    .addLayer("cnn55", conv3x3(192, 384, 0.2), "cnn53")
        	    .addLayer("cnn56", conv5x5(48, 128, 0.2), "cnn54")
        	    .addLayer("cnn57", conv1x1(832, 128, 0.2), "max13")
        	    .addVertex("depthconcat9", new MergeVertex(), "cnn52", "cnn55", "cnn56", "cnn57") // output: 7x7x1024
        	    .addLayer("avg3", avgPool7x7(1), "depthconcat9") // output: 1x1x1024
        	    .addLayer("fc1", fullyConnected(1024, 1024, 0.4), "avg3") // output: 1x1x1024
        	    .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        	      .nIn(1024)
        	      .nOut(outputNum)
        	      .activation("softmax")
        	      .build(), "fc1")
        	    .setOutputs("output")
        	    .backprop(true).pretrain(false)
        	    .build();
        
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }
}
