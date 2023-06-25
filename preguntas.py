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
    # Importe el stemmer de Porter
    # Importe CountVectorizer
    from ____ import ____

    # Cree un stemeer que use el algoritmo de Porter.
    stemmer = ____

    # Cree una instancia del analizador de palabras (build_analyzer)
    analyzer = ____().____()

    # Retorne el analizador de palabras
    return lambda x: (stemmer.stem(w) for w in analyzer(x))


def pregunta_04():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """

    # Importe CountVetorizer
    # Importe GridSearchCV
    # Importe Pipeline
    # Importe BernoulliNB
    from ____ import ____

    # Cargue las variables.
    x_train, _, y_train, _ = pregunta_02()

    # Obtenga el analizador de la pregunta 3.
    analyzer = pregunta_03()

    # Cree una instancia de CountVectorizer que use el analizador de palabras
    # de la pregunta 3. Esta instancia debe retornar una matriz binaria. El
    # límite superior para la frecuencia de palabras es del 100% y un límite
    # inferior de 5 palabras. Solo deben analizarse palabras conformadas por
    # letras.
    countVectorizer = ____(
        analyzer=____,
        lowercase=____,
        stop_words=____,
        token_pattern=____,
        binary=____,
        max_df=____,
        min_df=____,
    )

    # Cree un pipeline que contenga el CountVectorizer y el modelo de BernoulliNB.
    pipeline = ____(
        steps=[
            ("____", ____),
            ("____", ____()),
        ],
    )

    # Defina un diccionario de parámetros para el GridSearchCV. Se deben
    # considerar 10 valores entre 0.1 y 1.0 para el parámetro alpha de
    # BernoulliNB.
    param_grid = {
        "____": np.____(____, ____, ____),
    }

    # Defina una instancia de GridSearchCV con el pipeline y el diccionario de
    # parámetros. Use cv = 5, y "accuracy" como métrica de evaluación
    gridSearchCV = ____(
        estimator=____,
        param_grid=____,
        cv=____,
        scoring=____,
        refit=____,
        return_train_score=____,
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
    from ____ import ____

    # Obtenga el pipeline de la pregunta 3.
    gridSearchCV = pregunta_04()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el pipeline con los datos de entrenamiento usando la matriz de confusion.
    cfm_train = ____(
        y_true=____,
        y_pred=____.____(____),
    )

    cfm_test = ____(
        y_true=____,
        y_pred=____.____(____),
    )

    # Retorne la matriz de confusion de entrenamiento y prueba
    return cfm_train, cfm_test


def pregunta_06():
    """
    Pronóstico
    -------------------------------------------------------------------------------------
    """

    # Obtenga el pipeline de la pregunta 3.
    gridSearchCV = pregunta_04()

    # Cargue los datos generados en la pregunta 01.
    _, _, X_untagged, _ = pregunta_01()

    # pronostique la polaridad del sentimiento para los datos
    # no etiquetados
    y_untagged_pred = ____.____(____)

    # Retorne el vector de predicciones
    return y_untagged_pred
