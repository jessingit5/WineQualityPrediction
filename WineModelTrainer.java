package train;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.regression.LinearRegression;

public class WineModelTrainer {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Trainer")
                .getOrCreate();

        String filePath = "file:///home/ec2-user/WineQualityPrediction/TrainingDataset.csv";

        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(filePath);

        data = data.na().drop();
        System.out.println("Number of rows after dropping nulls: " + data.count());

        String[] featureCols = new String[]{
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide",
            "density", "pH", "sulphates", "alcohol"
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        LinearRegression lr = new LinearRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline().setStages(new org.apache.spark.ml.PipelineStage[]{assembler, lr});

        PipelineModel model = pipeline.fit(data);
        model.write().overwrite().save("file:///home/ec2-user/WineQualityPrediction/model");

        System.out.println("Model training complete and saved.");
        spark.stop();
    }
}
