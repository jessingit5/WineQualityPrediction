package predict;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.io.IOException;

public class WineModelPredictor {
    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: WineModelPredictor <path_to_test_csv>");
            System.exit(1);
        }

        String testFilePath = args[0];

        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Predictor")
                .getOrCreate();

        // Load the saved model
        LogisticRegressionModel model = LogisticRegressionModel.load("/home/ec2-user/WineQualityPrediction/wine_model");

        // Load the validation/test data
        Dataset<Row> testData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(testFilePath);

        // Prepare features
        String[] featureCols = testData.columns();
        String[] inputCols = java.util.Arrays.stream(featureCols)
                .filter(col -> !col.equals("quality"))
                .toArray(String[]::new);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> finalTestData = assembler.transform(testData)
                .select("features", "quality");

        // Make predictions
        Dataset<Row> predictions = model.transform(finalTestData);

        // Evaluate F1 score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);

        System.out.println("F1 Score = " + f1Score);

        spark.stop();
    }
}
