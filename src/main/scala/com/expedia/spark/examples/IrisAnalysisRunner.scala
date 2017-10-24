package com.expedia.spark.examples

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession

import scala.util.Random

object IrisAnalysisRunner extends App {

  case class Iris(sepalLength:  Float,
                   sepalWidth:   Float,
                   petalLength:  Float,
                   petalWidth:   Float,
                   species:      Int)

  case class StringyIris(sepalLength:  String,
                  sepalWidth:   String,
                  petalLength:  String,
                  petalWidth:   String,
                  species:      String)

  val PERCENT_TRAINING = 0.8

  val spark = SparkSession
    .builder()
    .appName("Iris")
    .master("local")
    .getOrCreate()

  import spark.implicits._

  val data = spark.read.option("header", true).csv("data/iris/iris.txt").as[StringyIris].map(sI => {
    val speciesCode = sI.species match {
      case "Iris-setosa" => 0
      case "Iris-versicolor" => 1
      case "Iris-virginica" => 2
    }
    Iris(sI.sepalLength.toFloat, sI.sepalWidth.toFloat, sI.petalLength.toFloat, sI.petalWidth.toFloat, speciesCode)
  })
  //yes I realize reading this in as a data frame and then collecting it is a bit pointless

  val collectedData = data.collect() //since it's tiny
  val mixedUpIndices = Random.shuffle(collectedData.indices.toList)
  val trainingIndices = mixedUpIndices.take((collectedData.length * PERCENT_TRAINING).toInt)
  val testIndices = mixedUpIndices.takeRight((collectedData.length * roundDouble(1 - PERCENT_TRAINING)).toInt)
  val trainingData = trainingIndices.collect(collectedData).toDF.as[Iris]
  val localTestData = testIndices.collect(collectedData)
  val testData = localTestData.toDF.as[Iris]

  val training = new VectorAssembler()
    .setInputCols(Array("sepalLength", "sepalWidth", "petalLength", "petalWidth"))
    .setOutputCol("features")
    .transform(trainingData)
    .withColumnRenamed("species", "label")

  // Load training data, and train the model with the training data.
  val model = new LogisticRegression().fit(training)

  // Prepare the test set and transform it into a Spark Dataset.
  val test = new VectorAssembler()
    .setInputCols(Array("sepalLength", "sepalWidth", "petalLength", "petalWidth"))
    .setOutputCol("features")
    .transform(testData)
    .select("features")

  // Predict labels for test set, and show this result.
  val predicted = model.transform(test)

  // compare
  val relevantColumns = predicted.select("features", "prediction")
  val rows = relevantColumns.collect()
  val comparable = rows.map(row => {
    (row.get(0).asInstanceOf[DenseVector].toArray.toList.map(roundDouble), row.get(1).toString.toFloat.toInt)
  }).toList //yuck !!! don't do this please

  val expected = localTestData.map(rec => {
    (List(
      roundDouble(rec.sepalLength.toDouble),
      roundDouble(rec.sepalWidth.toDouble),
      roundDouble(rec.petalLength.toDouble),
      roundDouble(rec.petalWidth.toDouble)), rec.species)
  })

  val compDF = comparable.toDF("numbers", "pred_species")
  val expDF = expected.toDF("numbers", "exp_species")

  val results = compDF.join(expDF, "numbers").select("pred_species", "exp_species")
  results.show()
  val collectedResults = results.collect()

  val same = collectedResults.count(r => r.get(0) == r.get(1)).toDouble/collectedResults.length.toDouble

  println(s"accuracy is ${same * 100}%")

  def roundDouble(d: Double) = {
    Math.round(d * 10.0) / 10.0
  }
}
