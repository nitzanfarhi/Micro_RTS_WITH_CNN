package ai.deeplearning;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class ImageClassify {
    // region static information
    static class MyLogger {
        void info(String s, Object... fmt) {
            System.out.println(String.format(s, fmt));
        }
    }
    private static final MyLogger log = new MyLogger();

    private static final String[] allowedExtensions = {
            "PassiveAI",
            "MouseController",
            "RandomAI",
            "RandomBiasedAI",
            "WorkerRush",
            "LightRush",
            "HeavyRush",
            "RangedRush",
            "CRush_V1",
            "POWorkerRush",
            "POLightRush",
            "POHeavyRush",
            "PORangedRush",
            "PortfolioAIv",
            "PGSAI",
            "IDRTMinimax",
            "IDRTMinimaxRandomized",
            "IDABCD",
            "MonteCarlov",
            "LSI",
            "UCT",
            "UCTUnitActions",
            "UCTFirstPlayUrgency",
            "NaiveMCTS",
            "BS3_NaiveMCTS",
            "MLPSMCTS",
            "AHTNAI",
            "InformedNaiveMCTS",
            "PuppetSearchMCTS",
            "PVAIML_ED"
    };
    private static final int outputNum = allowedExtensions.length;
    // endregion

    public static void main(String[] args) throws IOException {
        Random rnd = new Random(1239084);
        File parentDir = new File("C:\\Users\\guydan\\Desktop\\root_ai");

        System.out.println(parentDir);

        FileSplit filesInDir = new FileSplit(parentDir, BaseImageLoader.ALLOWED_FORMATS, rnd);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        System.out.println(labelMaker.getLabelForPath("C:\\Users\\guydan\\Desktop\\root_ai\\POHeavyRush\\1359699263145375.jpg"));
        BalancedPathFilter pathFilter = new BalancedPathFilter(rnd, BaseImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageRecordReader trainRecordReader = new ImageRecordReader(32,32,1,labelMaker);
        trainRecordReader.initialize(trainData);
        System.out.println(trainData.length());

        ImageRecordReader testRecordReader = new ImageRecordReader(32,32,1,labelMaker);
        testRecordReader.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, 10, 1, outputNum);
//        DataNormalization trainScaler = new ImagePreProcessingScaler(0,1);
//        trainScaler.fit(trainIter);
//        trainIter.setPreProcessor(trainScaler);

        DataSetIterator testIter = new RecordReaderDataSetIterator(trainRecordReader, 10, 1, outputNum);
//        DataNormalization testScaler = new ImagePreProcessingScaler(0,1);
//        testScaler.fit(trainIter);
//        trainIter.setPreProcessor(testScaler);

        int seed = 9090;
        int iterations = 64;
        // region ann config
        int nChannels = 1;

        // learning rate schedule in the form of <Iteration #, Learning Rate>
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.05)//.biasLearningRate(0.02)
                /*
                    Alternatively, you can use a learning rate schedule.

                    NOTE: this LR schedule defined here overrides the rate set in .learningRate(). Also,
                    if you're using the Transfer Learning API, this same override will carry over to
                    your new model configuration.
                */
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                /*
                    Below is an example of using inverse policy rate decay for learning rate
                */
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse)
                //.lrPolicyDecayRate(0.001)
                //.lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS) //To configure: .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(32,32,1)) //See note below
                .backprop(true).pretrain(false).build();
        // endregion

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        int nEpochs = 50;
        // region evaluation
        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i < nEpochs; i++ ) {
            model.fit(trainIter);
            log.info("Completed epoch %d", i);

            log.info("Evaluate model....");
            Evaluation eval = model.evaluate(testIter);
            log.info(eval.stats());
            testIter.reset();
        }
        log.info("****************Example finished********************");
        //endregion
    }
}
