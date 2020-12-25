import os

from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import NGram, VectorAssembler, CountVectorizer, IDF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vector, Vectors, VectorUDT
# from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, FloatType, ArrayType, StringType
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import FMClassifier
from pyspark.sql.window import Window as W
from spacy.lang.en import English
import spacy
from pyspark.ml.stat import Summarizer
from collections import Counter
import pandas
import time

nlp = spacy.load("en_core_web_sm")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6"
conf = SparkConf().setAppName('Spark DL Tabular Pipeline').setMaster('local[6]').set('spark.driver.memory', '16g').set(
    'spark.executor.memory', '6g')

print(conf.getAll())
sc = SparkContext(conf=conf)
sql_context = SparkSession(sc)



# Load Data to Spark Dataframe
df = sql_context.read.csv('final/amazon_reviews.tsv',
                          header=True,
                          sep=r'\t',
                          inferSchema=True)

tags = ['SYM', 'PUNCT', 'X', 'ADJ', 'CCONJ', 'NUM', 'DET', 'PRON', 'ADP', 'ADJ', 'VERB', 'NOUN', 'PROPN', 'ADV', 'SPACE', 'PART', 'INTJ', 'AUX', 'SCONJ']
def spacy_pos(text):
    doc = nlp(text)
    c = Counter([token.pos_ for token in doc])
    sbase = sum(c.values())
    result = [0] * 19
    for el, cnt in c.items():
        index = tags.index(el)
        result[index] = (cnt / sbase)
    return Vectors.dense(result).asML()


def get_spacy_udf():
    def get_spacy(text):
        global nlp
        try:
            doc = nlp(text)
            c = Counter([token.pos_ for token in doc])
            sbase = sum(c.values())
        except:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            c = Counter([token.pos_ for token in doc])
            sbase = sum(c.values())
        return [count for count in sbase]
    res_udf = udf(get_spacy, StringType(ArrayType()))
    return res_udf


# Convert to Lower Case, remove html tags
def clean_text(c):
    c = lower(c)
    c = regexp_replace(c, "<.*?>", "")
    return c


clean_text_df = df.select(clean_text(col("REVIEW_TEXT")).alias("REVIEW_TEXT"), "LABEL", "RATING", "VERIFIED_PURCHASE",
                          "DOC_ID", "PRODUCT_CATEGORY")

tokenizer = Tokenizer(inputCol="REVIEW_TEXT", outputCol="words")

countTokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(clean_text_df)
tokenized.select("REVIEW_TEXT", "words", "LABEL", "PRODUCT_CATEGORY") \
    .withColumn("tokens", countTokens(col("words")))#.show(truncate=False)


tokenized.withColumn('pos', col('REVIEW_TEXT'))
print('SPACY')
review_list = tokenized.select('REVIEW_TEXT').rdd.flatMap(lambda x: x).collect()
review_list = [spacy_pos(item) for item in review_list]
# print(review_list)
pos_df = sql_context.createDataFrame([(l,) for l in review_list], ['pos'])
pos_df = pos_df.withColumn("idx", F.monotonically_increasing_id())
tokenized = tokenized.withColumn("idx", F.monotonically_increasing_id())
windowSpec = W.orderBy("idx")
pos_df = pos_df.withColumn("idx", F.row_number().over(windowSpec))
tokenized = tokenized.withColumn("idx", F.row_number().over(windowSpec))
tokenized = tokenized.join(pos_df, tokenized.idx == pos_df.idx).drop("idx")


# print()
# print('----------------')
# ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
#
# ngramDataFrame = ngram.transform(tokenized)
# ngramDataFrame.select("REVIEW_TEXT", "ngrams", "LABEL") \
#     .withColumn("tokens", countTokens(col("ngrams")))#.show(truncate=False)


# Used for lambda unpacking
def star(f):
    return lambda args: f(*args)


def featurize(word_column="words", vectorizer="w2v"):
    print('Featurizing...')
    if vectorizer == "w2v":
        embedder = Word2Vec(vectorSize=100, minCount=0, inputCol=word_column, outputCol="embeddings")

    else:
        start = time.time()
        cv = CountVectorizer(inputCol=word_column, outputCol="rawembeddings")
        embedder = IDF(inputCol="rawembeddings", outputCol="embeddings")


    si_verified = StringIndexer(inputCol="VERIFIED_PURCHASE", outputCol="verified")
    si_category = StringIndexer(inputCol="PRODUCT_CATEGORY", outputCol="category")
    assembler = VectorAssembler(
        inputCols=["embeddings", "RATING", "verified", 'pos', "category"],
        outputCol="features")
    if vectorizer == "w2v":
        doc2vec_pipeline = Pipeline(stages=[embedder, si_verified, si_category, assembler])
    else:
        doc2vec_pipeline = Pipeline(stages=[cv, embedder, si_verified, si_category, assembler])
    if word_column == "ngrams":
        doc2vec_model = doc2vec_pipeline.fit(ngramDataFrame)
        doc2vecs_df = doc2vec_model.transform(ngramDataFrame)
    else:
        start = time.time()
        doc2vec_model = doc2vec_pipeline.fit(tokenized)
        doc2vecs_df = doc2vec_model.transform(tokenized)
        end = time.time()
        print('ELAPSED TIME:')
        print(end - start)
        print('With Pos')
        print(doc2vecs_df.take(10))
    return doc2vecs_df


def evaluate(model, word_column="words", vectorizer="w2v"):
    doc2vecs_df = featurize(word_column, vectorizer)
    if type(model) == LinearSVC:
        paramGrid = ParamGridBuilder() \
            .addGrid(model.regParam, [0.1]) \
            .build()
    elif type(model) == GBTClassifier:
        paramGrid = ParamGridBuilder() \
            .addGrid(model.maxIter, [50]) \
            .build()
    elif type(model) == RandomForestClassifier:
        paramGrid = ParamGridBuilder() \
            .addGrid(model.maxBins, [100]) \
            .build()
    elif type(model) == MultilayerPerceptronClassifier:
        paramGrid = ParamGridBuilder() \
             .addGrid(model.layers, [[122, 50, 2]]) \
             .build()
        # .addGrid(model.layers, [[120, 2], [120, 50, 2], [120, 75, 50, 2]]) \
    elif type(model) == FMClassifier:
        paramGrid = ParamGridBuilder() \
            .addGrid(model.stepSize, [.01, .001]) \
            .build()
    print('Evaluating...')
    w2v_train_df, w2v_test_df = doc2vecs_df.randomSplit([0.8, 0.2])
    si = StringIndexer(inputCol="LABEL", outputCol="label")
    model_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    classifier_pipeline = Pipeline(stages=[si, model])
    crossval = CrossValidator(estimator=classifier_pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=model_evaluator,
                              numFolds=5)
    fit_model = crossval.fit(doc2vecs_df)
    predictions = fit_model.transform(w2v_test_df)
    # predictions.toPandas().to_csv('predictions.csv')
    # predictions.groupBy('prediction', 'label', 'PRODUCT_CATEGORY')
    # predictions.describe()
    summarizer = Summarizer.metrics("mean", "count")
    predictions.select(summarizer.summary(predictions.filter(predictions.label == 1).pos)).show(truncate=False)
    preds_and_labels = predictions.select(['prediction', 'label'])
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    print('Confusion Matrix')
    print(metrics.confusionMatrix().toArray())
    # Overall statistics
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)
    accuracy = model_evaluator.evaluate(predictions)
    trainingSummary = fit_model.bestModel.stages[-1].extractParamMap()
    print(trainingSummary)

    return accuracy


def models():
    rf_classifier = RandomForestClassifier(labelCol="label", featuresCol="features")
    print("Random Forest F1 = %g" % evaluate(rf_classifier))
    lsvc = LinearSVC(maxIter=50)
    print("Linear SVC F1 = %g" % evaluate(lsvc))
    gbt = GBTClassifier()
    print("GBT F1 = %g" % evaluate(gbt))

    mlp = MultilayerPerceptronClassifier(seed=1234, featuresCol='features')
    print("MLP F1 = %g" % evaluate(mlp))

    fm = FMClassifier()
    print('FM')
    evaluate(fm)
    featurize_lda()
    # NGrams
    # print("NGram Random Forest F1 = %g" % evaluate(rf_classifier, "ngrams"))
    # print("Ngram Linear SVC F1 = %g" % evaluate(lsvc, "ngrams"))
    # print("Ngram GBT F1 = %g" % evaluate(gbt, "ngrams"))
    # TF-IDF
    print("Ngram TF-IDF Random Forest F1 = %g" % evaluate(rf_classifier, "ngrams", "TF-IDF"))
    print("Ngram TF-IDF Linear SVC F1 = %g" % evaluate(lsvc, "ngrams", "TF-IDF"))
    print("Ngram TF-IDF GBT F1 = %g" % evaluate(gbt, "ngrams", "TF-IDF"))
    print("Words TF-IDF Random Forest F1 = %g" % evaluate(rf_classifier, "words", "TF-IDF"))
    print("Words TF-IDF Linear SVC F1 = %g" % evaluate(lsvc, "words", "TF-IDF"))
    print("Words TF-IDF GBT F1 = %g" % evaluate(gbt, "words", "TF-IDF"))


models()
