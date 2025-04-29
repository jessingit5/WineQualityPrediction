package train;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import java.io.IOException; 

public class WineModelTrainer {
    public static void main(String[] args) throws IOException { // ✅ ADD throws IOException
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Trainer")
                .getOrCreate();

        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("/home/ec2-user/WineQualityPrediction/TrainingDataset.csv");

        String[] featureCols = data.columns();
        String[] inputCols = java.util.Arrays.stream(featureCols)
                .filter(col -> !col.equals("quality"))
                .toArray(String[]::new);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> output = assembler.transform(data)
                .select("features", "quality");

        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features");

        LogisticRegressionModel model = lr.fit(output);

        model.save("/home/ec2-user/WineQualityPrediction/wine_model"); // This line can throw IOException

        spark.stop();
    }
}
