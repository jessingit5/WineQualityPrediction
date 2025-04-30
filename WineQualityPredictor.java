package predict;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.PipelineModel;

public class WineQualityPredictor {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Predictor")
                .getOrCreate();

        String testFile = "file:///home/ec2-user/WineQualityPrediction/ValidationDataset.csv";

        Dataset<Row> testData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(testFile);

        testData = testData.na().drop();

        PipelineModel model = PipelineModel.load("file:///home/ec2-user/WineQualityPrediction/model");

        Dataset<Row> predictions = model.transform(testData);
        predictions.select("features", "quality", "prediction").show(10);

        spark.stop();
    }
}
