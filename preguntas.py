"""
Análisis de Sentimientos usando Naive Bayes
-----------------------------------------------------------------------------------------

El archivo `amazon_cells_labelled.txt` contiene una serie de comentarios sobre productos
de la tienda de amazon, los cuales están etiquetados como positivos (=1) o negativos (=0)
o indterminados (=NULL). En este taller se construirá un modelo de clasificación usando
Naive Bayes para determinar el sentimiento de un comentario.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """
 # Lea el archivo `amazon_cells_labelled.tsv` y cree un DataFrame usando pandas.
    # Etiquete la primera columna como `msg` y la segunda como `lbl`. Esta función
    # retorna el dataframe con las dos columnas.
    df = pd.read_csv(
        "amazon_cells_labelled.tsv",
        sep="\t",
        header=None,
        names=["msg", "lbl"],
    )
    #print(df)
    
    df_tagged =df[df["lbl"].notnull()]
    df_untagged = df[df["lbl"].isnull()]

    x_tagged = df_tagged["msg"]
    y_tagged = df_tagged["lbl"]

    x_untagged = df_untagged["msg"]
    y_untagged = df_untagged["lbl"]

    
    return x_tagged, y_tagged, x_untagged, y_untagged


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """


    from sklearn.model_selection import train_test_split
   

    
    x_tagged, y_tagged, _, _ = pregunta_01()
    #en la linea 55 Cargue los datos generados en la pregunta 01.
    

    x_train, x_test, y_train, y_test = train_test_split(
        x_tagged,
        y_tagged,
        test_size=0.1,
        #aca estamos utilizando el 10% de los datos y en la linea de abajo estamos definiendo la semilla 12345
        random_state=12345,
    )

    return x_train, x_test, y_train, y_test

def pregunta_03():
    """
    Construcción de un analizador de palabras
    -------------------------------------------------------------------------------------
    """
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer

    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()
    return lambda x: (stemmer.stem(w) for w in analyzer(x))


def pregunta_04():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """
#primera parte / importancion de liberias necesarias para los datos que vamos a utilizar 
     # Importe CountVetorizer
    from sklearn.feature_extraction.text import CountVectorizer
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # Importe GridSearchCV
    from sklearn.model_selection import GridSearchCV
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # Importe Pipeline
    from sklearn.pipeline import Pipeline
    #https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    # Importe BernoulliNB
    from sklearn.naive_bayes import BernoulliNB
    #https: // scikit - learn.org / stable / modules / generated / sklearn.naive_bayes.BernoulliNB.html
     # variables.
    x_train, _, y_train, _ = pregunta_02()

    #analizador de la pregunta 3.
    analyzer = pregunta_03()

    # matriz par aanalizar palabras conformadas por letras
    countVectorizer = CountVectorizer(
        analyzer=analyzer,
        lowercase=True,
        stop_words="english",
        token_pattern=r"\b\w\w+\b",
        binary=True,
        max_df=1.0,
        min_df=5,
    )

    # pipeline para Bernoulli 
    pipeline = Pipeline(
        steps=[
            ("CountVectorizer", countVectorizer),
            ("BernoulliNB", BernoulliNB()),
        ],
    )

    # diccionario
    param_grid = {
        "BernoulliNB__alpha": np.linspace(0.1,1,10)
    }

    # Defina una instancia de GridSearchCV  (biblioteca de aprendizaje automático) busca combinaciones posibles de hiper parametros 
    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        refit=True,
        return_train_score=False,
    )

    # Búsque la mejor combinación de regresores
    gridSearchCV.fit(x_train, y_train)

    # Retorne el mejor modelo
    return gridSearchCV



def pregunta_05():
    """
    Evaluación del modelo
    -------------------------------------------------------------------------------------
    """

    # Importe confusion_matrix
    from sklearn.metrics import confusion_matrix
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    #pipeline de la pregunta 4.
    gridSearchCV = pregunta_04()

    #variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el pipeline con los datos de entrenamiento usando la matriz de confusion.
    cfm_train = confusion_matrix(
        y_true=y_train,
        y_pred=gridSearchCV.predict(X_train),
    )

    cfm_test = confusion_matrix(
        y_true=y_test,
        y_pred=gridSearchCV.predict(X_test),
    )

    # Retorne la matriz de confusion de entrenamiento y prueba
    return cfm_train, cfm_test


def pregunta_06():
    """
    Pronóstico
    -------------------------------------------------------------------------------------
    """

    #pipeline de la pregunta 4.
    gridSearchCV = pregunta_04()

    # datos generados en la pregunta 01.
    _, _, X_untagged, _ = pregunta_01()

    # pronostico
    # no etiquetados
    y_untagged_pred = gridSearchCV.predict(X_untagged)

    #vector de predicciones
    return y_untagged_pred