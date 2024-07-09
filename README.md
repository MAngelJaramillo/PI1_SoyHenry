![henry](https://github.com/GRP-777/Proyecto_Individual_1/assets/132501854/1333fbec-6c93-4f2d-a1ff-6b1ad899051c)

# Henry Proyecto Individual 1
# Data Science & Machine Learning Operations (MLOps)

![MLOps](https://github.com/GRP-777/Proyecto_Individual_1/assets/132501854/c5259852-e96b-439c-a1af-f89124128043)

# Introducción

Este proyecto está basado en datos ficticios y no en busca de un analisis verdadero sino para demostrar las habilidades aprendidas durante el Henry Bootcamp.

Se me asigno la tarea de desarrollar un _**Minimum Viable Product (MVP)**_ para desarrollar una API utlizando **FastAPI** que contiviese 6 funciones con sus respectivos endpoints y uno extra como sistema de recomendaciones utilizando machine learning.

![PI1_MLOps_Mapa1](https://github.com/GRP-777/Proyecto_Individual_1/assets/132501854/f36720bf-8322-48a0-a002-95dd2acc1944)


# Dataset Descripción y Diccionario
Para descargar los datasets originales se puede acceder desde el siguiente link: [Original Datasets](https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5)


| **Columna**     | **Descripcion**                                              | **Ejemplo**                                                                                                                                                                |
|------------------- |------------------------------------------------------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| adult          | Indica si la película tiene calificación X, exclusiva para adultos.                                            | FALSE                                                                                                                                           |
| belongs_to_collection             | Un diccionario que indica a qué franquicia o serie de películas pertenece la película. Incluye id, name, poster_path y backdrop_path.                                                     | {'id': 10194, 'name': 'Toy Story Collection', 'poster_path': '/7G9915LfUQ2lVfwMEEhDsn3kT4B.jpg', 'backdrop_path': '/9FBwqcd9IRruEDUrTdcaafOMKUq.jpg'}                                                                                                                                    |
| budget       | El presupuesto de la película, en dólares.                                                      | 30000000                                                                                                                                                 |
| genres             | Un diccionario que indica todos los géneros asociados a la película. Incluye id y name.                                                      | "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]"                                                                         |
| homepage                | La página web oficial de la película.                                                                                                                     | http://toystory.disney.com/toy-story
| id| ID de la película en la base de datos.                                                    | 862                                                                                                                                                                        |
| imdb_id               | IMDB ID de la película.                                                       | tt0114709                                                                                       |
| original_language     | Idioma original en el que se grabó la película.                                                     | en                                                                                                                                                          |
| original_title       | Título original de la película.                                                | Toy Story                                                                                                           |
| overview              | Pequeño resumen de la película/Sinopsis                                         | "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences."                                                                                                              |
| popularity              | Puntaje de popularidad de la película, asignado por TMDB (TheMoviesDataBase).                                                      | 21.946943                                                                                                                                             |
| poster_path       | URL del póster de la película.                                                       | /rhIRbceoE9lR4veEXuwCC2wARtG.jpg                                                                                                                                                                        |
| production_companies                 | Lista con las compañías productoras asociadas a la película. Incluye name e id.                                          | [{'name': 'Pixar Animation Studios', 'id': 3}]"                                                                                                                                                           |
| production_countries          | Lista con los países donde se produjo la película. Incluye iso_3166_1 y name.                                                         | "[{'iso_3166_1': 'US', 'name': 'United States of America'}]"                                                                                                                                        |
| release_date         | Fecha de estreno de la película.                                                   | 1995-10-30                                                                                                                                                                  |
| revenue            | Recaudación de la película, en dólares.                                             | 373554033                                                                                                                                         |
| runtime          | Duración de la película, en minutos.                                                   | 81                                                                                                                                         |
| status           | Estado actual de la película (si fue anunciada, si ya se estrenó, etc).                                        | Released                                     |                                                                                                                                                                                   |
| spoken_languages            | Lista con los idiomas que se hablan en la película. Incluye iso_639_1 y name.                                             | "[{'iso_639_1': 'en', 'name': 'English'}]"                                                                                                                                        |
| tagline           | Frase célebre asociada a la película.                                                  | ""                                                                                                                                             |
| title              | Título de la película.                                         | Toy Story                                                                                |}
| video | Indica si hay o no un tráiler en video disponible en TMDB. | FALSE
|vote_average | Puntaje promedio de reseñas de la película. | 7.7
|vote_count | Número de votos recibidos por la película en TMDB. | 5415
| cast | Una lista de diccionarios que representa los actores del reparto de la película. Cada diccionario contiene información detallada sobre cada miembro del reparto.
| crew | Una lista de diccionarios que representa los miembros del equipo de producción de la película. Cada diccionario contiene información detallada sobre cada miembro del equipo.

Procesos

_**ETL**_:
Para aprender más sobre el proceso de limpieza de datos por favor acceder al documento adjunto en este repositorio de nombre "ETL.ipynb"

A continuación se hara un resumen del proceso de ETL aún así:

_**Nombres de los datasets**_:
- movies_dataset
- credits

_**Desanidado**_:
El primer tratamiento que se hizo a los datos fue un proceso de desanidado para distintas columnas que contenian valores en forma de diccionarios, listas o incluso diccionarios de listas.

_**Drop de columnas inutilizadas**_:

Se tuvo que hacer un drop a las columnas de los datasets que no fueran utiles, esto se explica más a detalle en el notebook EDA_PI1 y al final del notebook ETL.


_**Control de valores nulos**_:

Se hizo un control de nulos a las columnas de "revenue", "budget" y "release date", todas del dataset de movies.

_**Arreglo de Daytime datatype**_:

Se hizo un cambio al datatype de la columna "realease_date" para que fuese tratado como un Daytime datatype (AAAA-mm-dd)

_**Meging de los datasets**_:

Se hizo un merge a ambos datasets para poder trabajarlos mas comodos durante el desarrollo de la API.

_**Columna de retorno**_:

Se creo una nueva columna con el retorno de inversión, llamada return con los campos revenue y budget, dividiendo estas dos últimas revenue / budget"

# _Funciones_
- Para mas detalle sobre el desarrollo de las funciones para la API revisar el archivo "main.py" de este repositorio.

Desarrollo API: Se propone disponabilizar los datos de una empresa usando el framework FastAPI.

Se crearon 6 funciones para los endpoints que se consumirán en la API.

1.def cantidad_filmaciones_mes( Mes ): Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.
                    Ejemplo de retorno: X cantidad de películas fueron estrenadas en el mes de X

2.def cantidad_filmaciones_dia( Dia ): Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.
                    Ejemplo de retorno: X cantidad de películas fueron estrenadas en los días X

3.def score_titulo( titulo_de_la_filmación ): Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
                    Ejemplo de retorno: La película X fue estrenada en el año X con un score/popularidad de X

4.def votos_titulo( titulo_de_la_filmación ): Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.
                    Ejemplo de retorno: La película X fue estrenada en el año X. La misma cuenta con un total de X valoraciones, con un promedio de X

5.def get_actor( nombre_actor ): Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, la cantidad de películas que en las que ha participado y el promedio de retorno. La definición no deberá considerar directores.
                    Ejemplo de retorno: El actor X ha participado de X cantidad de filmaciones, el mismo ha conseguido un retorno de X con un promedio de X por filmación

5.def get_director( nombre_director ): Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.

# _**Machine Learning**_
Por último se desarrollo un modelo de machine learning utilizando la similitud de coseno de la librería scikit-learn cuya función busca la similitud entre el score (determinado por el vote_average) de la pelicula ingresada al sistema y la compara entre las 5 más similares y arroja al usuario las recomendaciones.

# _**API Deployment**_
El deployement de esta API se puede encontrar en el siguente link para poder acceder a todos sus endpoints:

[API Deployment]()

# Requisitos
- Python
- Regular Expression Operations
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- FastAPI
- [Render](https://render.com/)

# _Autor_
- Miguel Ángel Jaramillo
- Mail: miguelangelgomezj@gmail.com
