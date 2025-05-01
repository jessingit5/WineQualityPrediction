package predict; // Or your actual package name

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class WineQualityPredictor {

    public static void main(String[] args) {

        // --- Argument Handling ---
        if (args.length != 2) {
            System.err.println("Usage: WineQualityPredictor <modelPath> <inputDataPath>");
            System.err.println("Example: WineQualityPredictor file:///.../saved_wine_model file:///.../ValidationDataset.csv");
            System.exit(1);
        }
        String modelPath = args[0];
        String inputDataPath = args[1]; // Path to the input data CSV (must include header)

        // --- 1. Initialize Spark Session ---
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Predictor (With Headers)")
                .getOrCreate();

        System.out.println("Spark Session Initialized for Prediction.");
        System.out.println("Loading model from: " + modelPath);
        System.out.println("Loading data to predict from: " + inputDataPath + " (Ensure includes header row)");

        // --- 2. Define Schema with Actual Header Names ---
        // MUST match the schema used during training
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


        // --- 3. Load Model ---
        System.out.println("Loading model...");
        PipelineModel loadedModel = PipelineModel.load(modelPath);
        System.out.println("PipelineModel loaded successfully.");

        // --- 4. Load Input Data (Expecting Header Row) ---
        System.out.println("Loading input data (with header)...");
         Dataset<Row> inputData = spark.read()
                .option("header", "true") // Use the header row in the file
                .option("delimiter", ";")
                .schema(schema) // Apply defined schema
                .csv(inputDataPath)
                .withColumnRenamed("quality", "label") // Rename quality column for evaluator
                .na().drop();

        System.out.println("Input Data Schema after loading:");
        inputData.printSchema();
        long inputCount = inputData.count();
         System.out.println("Input data rows count: " + inputCount);

        if (inputCount == 0) {
             System.err.println("Error: Input data is empty. Check path and file contents (including header).");
             spark.stop();
             System.exit(1);
        }

        // --- 5. Make Predictions ---
        // The loaded pipeline model applies the VectorAssembler (which expects the actual header names)
        System.out.println("Making predictions...");
        Dataset<Row> predictions = loadedModel.transform(inputData);

        System.out.println("Predictions sample:");
        // Show actual label ("quality" renamed to "label") and predicted label
        predictions.select("label", "prediction").show(20);

        // --- 6. Evaluate (Calculate F1 Score) ---
        // This assumes the input data CSV contains the true "quality" column
        System.out.println("Evaluating F1 Score...");
         MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        System.out.println("----------------------------------------");
        System.out.println("F1 Score on Input Data = " + f1Score);
        System.out.println("----------------------------------------");

        // --- 7. Stop Spark Session ---
        System.out.println("Stopping Spark session.");
        spark.stop();
    }
}