package train; // Or your actual package name

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;

public class WineModelTrainer {

    public static void main(String[] args) throws IOException {

        // --- Argument Handling ---
        if (args.length != 3) {
            System.err.println("Usage: WineModelTrainer <trainingCsvPath> <validationCsvPath> <modelSavePath>");
            System.err.println("Example: WineModelTrainer file:///.../TrainingDataset.csv file:///.../ValidationDataset.csv file:///.../saved_wine_model");
            System.exit(1);
        }
        String trainingDataPath = args[0];
        String validationDataPath = args[1];
        String modelSavePath = args[2];

        // --- 1. Initialize Spark Session ---
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Trainer (With Headers)")
                .getOrCreate();

        System.out.println("Spark Session Initialized.");
        System.out.println("Training Data Path: " + trainingDataPath + " (Ensure includes header row)");
        System.out.println("Validation Data Path: " + validationDataPath + " (Ensure includes header row)");
        System.out.println("Model Save Path: " + modelSavePath);

        // --- 2. Define Schema with Actual Header Names ---
        // This defines the expected structure and types.
        StructType schema = new StructType(new StructField[]{
                DataTypes.createStructField("fixed acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("volatile acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("citric acid", DataTypes.DoubleType, true),
                DataTypes.createStructField("residual sugar", DataTypes.DoubleType, true),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, true),
                DataTypes.createStructField("free sulfur dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("total sulfur dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("density", DataTypes.DoubleType, true),
                DataTypes.createStructField("pH", DataTypes.DoubleType, true),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, true),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, true),
                DataTypes.createStructField("quality", DataTypes.DoubleType, true) // Target label
        });

        // --- 3. Load Data (Expecting Header Row) ---
        System.out.println("Loading training data (with header)...");
        Dataset<Row> trainingData = spark.read()
                .option("header", "true") // Use the header row in the file
                .option("delimiter", ";")
                .schema(schema) // Apply defined schema (recommended over inferSchema)
                .csv(trainingDataPath)
                .withColumnRenamed("quality", "label") // Rename target column
                .na().drop();

        System.out.println("Loading validation data (with header)...");
         Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .schema(schema)
                .csv(validationDataPath)
                .withColumnRenamed("quality", "label")
                .na().drop();

        System.out.println("Training data schema after loading:");
        trainingData.printSchema();
        System.out.println("Validation data schema after loading:");
        validationData.printSchema();
        long trainCount = trainingData.count();
        long valCount = validationData.count();
        System.out.println("Training rows count: " + trainCount);
        System.out.println("Validation rows count: " + valCount);

        if (trainCount == 0 || valCount == 0) {
             System.err.println("Error: Training or validation data is empty. Check paths and file contents (including headers).");
             spark.stop();
             System.exit(1);
        }

        // --- 4. Prepare Features (using actual header names) ---
        String[] featureCols = {
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"
                // Excludes "quality" (which is now "label")
        };
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        // --- 5. Define Classification Model (Logistic Regression) ---
        System.out.println("Defining Logistic Regression model...");
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setMaxIter(20)
                .setRegParam(0.05);

        // --- 6. Create Pipeline ---
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, lr});

        // --- 7. Train Model ---
        System.out.println("Starting model training...");
        PipelineModel trainedModel = pipeline.fit(trainingData);
        System.out.println("Model training completed.");

        // --- 8. Evaluate on Validation Data ---
        System.out.println("Making predictions and evaluating on validation data...");
        Dataset<Row> predictions = trainedModel.transform(validationData);

        System.out.println("Validation Predictions Sample:");
        predictions.select("features", "label", "prediction").show(10, false);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        System.out.println("----------------------------------------");
        System.out.println("F1 Score on Validation Data = " + f1Score);
        System.out.println("----------------------------------------");

        // --- 9. Save Model ---
        System.out.println("Saving trained pipeline model to: " + modelSavePath);
        trainedModel.write().overwrite().save(modelSavePath);
        System.out.println("Model saved successfully.");

        // --- 10. Stop Spark Session ---
        System.out.println("Stopping Spark session.");
        spark.stop();
    }
}