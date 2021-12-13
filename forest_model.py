from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


#create spark session
spark = SparkSession.builder.appName('cluster').getOrCreate()

#loading data
df = spark.read.csv('million.csv', inferSchema=True, header=True)

#showing data
#df.show(3)
#showing features
#print out the feature name
#|index|label|blue_min|blue_max|blue_median|blue_avminmax|AMP_blue_max_blue_min|green_min|green_max|green_median|green_avminmax|AMP_green_max_green_min|red_min|red_max|red_median|red_avminmax|AMP_red_max_red_min|nir_min|nir_max|nir_median|nir_avminmax|AMP_nir_max_nir_min|swir1_min|swir1_max|swir1_median|swir1_avminmax|AMP_swir1_max_swir1_min|swir2_min|swir2_max|swir2_median|swir2_avminmax|AMP_swir2_max_swir2_min|RN_min|RN_max|RN_median|RN_avminmax|AMP_RN_max_RN_min|flag|

#df.printSchema()
#print(df.count())
#dont nee index and label for features
feature_columns = df.columns[2:]

#Creating a features array
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features") 
#use the assembler to create the features column
data = assembler.transform(df) 

#split training/testing dataset
train,test = data.randomSplit([0.7,0.3])
#train,test = data.randomSplit([0.75,0.25])

#apply classifier to the dataset
#model selected
#a. logistic regression
def logistic_regression(cv=True):
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.1, featuresCol='features',labelCol='label')
    if not(cv):
        model = lr.fit(train) 
        #evaluation_summary = model.evaluate(test) 
        #predictions = model.transform(test)
        predictions = model.transform(test)

        #take a look on the predictions
        #print(predictions.take(1))

        #getting scores
        modelname = 'logistic regression'
        show_results(predictions, modelname)

    else:
        # Create ParamGrid for Cross Validation
        grid = (ParamGridBuilder()
                    .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
                    .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
                    .build())
        evaluator = BinaryClassificationEvaluator()
        # Create k-fold CrossValidator, k=5
        cv = CrossValidator(estimator=lr,
                            estimatorParamMaps=grid,
                            evaluator=evaluator,
                            numFolds=5)
        cvModel = cv.fit(train)

        predictions = cvModel.transform(test)
        # Evaluate best model
        #getting scores
        #evaluator = MulticlassClassificationEvaluator(
        #    labelCol="label", predictionCol="prediction", metricName="accuracy")
        #accuracy = evaluator.evaluate(predictions)
        #print(accuracy)
        modelname = 'Logistic regression, 5-fold'
        show_results(predictions, modelname)
    

#Random Forest
def Random_Forest(cv = True):
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    if not (cv):
        model = rf.fit(train)
        evaluation_summary = model.evaluate(test) 
        predictions = model.transform(test)

        #getting scores
        #evaluator = MulticlassClassificationEvaluator(
        #    labelCol="label", predictionCol="prediction", metricName="accuracy")
        #accuracy = evaluator.evaluate(predictions)
        modelname = 'Random Forest'
        show_results(predictions, modelname)
    else:
        #Random Forest Cross Validation of 5-fold
        grid = (ParamGridBuilder().addGrid(rf.numTrees, [1, 3, 5])
                                .addGrid(rf.maxDepth, [3, 5, 7, 10])
                                .addGrid(rf.maxBins, [20, 30, 40])
                                .build())
   
        evaluator = BinaryClassificationEvaluator()
        cv = CrossValidator(estimator=rf,
                    evaluator=evaluator,
                    estimatorParamMaps=grid,
                    numFolds=5)
        cvModel_rf = cv.fit(train)

        predictions = cvModel_rf.transform(test)
        #accuracy = evaluator.evaluate(predictions)
        #print("Random Forest with 5-fold Acc = %g" % (accuracy))
        
        modelname = 'Random Forest, 5-fold'
        show_results(predictions, modelname)
        
#Best model on all data and plot map
def best():
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    
    #Random Forest Cross Validation of 5-fold
    grid = (ParamGridBuilder().addGrid(rf.numTrees, [1, 3, 5])
                            .addGrid(rf.maxDepth, [3, 5, 7, 10])
                            .addGrid(rf.maxBins, [20, 30, 40])
                            .build())
   
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=rf,
                evaluator=evaluator,
                estimatorParamMaps=grid,
                numFolds=5)
    cvModel_rf = cv.fit(train)

    predictions = cvModel_rf.transform(test)
    accuracy = evaluator.evaluate(predictions)
    print("Random Forest best acc = %g" % (accuracy))
    predictions = cvModel_rf.transform(data)
    plot_map(predictions)
    
def plot_map(prdct):
    y_true = prdct.select(['label']).collect()
    y_pred = prdct.select(['prediction']).collect()
    forest_image_true = np.reshape(y_true, (-1, 1000))
    forest_image_pred = np.reshape(y_pred, (-1, 1000))
    
    plt.matshow(forest_image_true)
    plt.xlabel("pixel")
    plt.ylabel("pixel")
    plt.show()
    
    plt.matshow(forest_image_pred)
    plt.xlabel("pixel")
    plt.ylabel("pixel")
    plt.show()
    
def show_results(prdct, modelname):
    y_true = prdct.select(['label']).collect()
    y_pred = prdct.select(['prediction']).collect()
    print(classification_report(y_true, y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix = conf_matrix/ conf_matrix.astype(np.float).sum(axis=1)
    conf_matrix = np.around(conf_matrix, decimals=3)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title('Normalized Confusion Matrix_'+modelname, fontsize=14)
    plt.show()

    
if __name__=="__main__":
    print('==========run here===========')
    #logistic_regression(False)
    #logistic_regression()
    #Random_Forest(False)
    #Random_Forest()
    #best()
