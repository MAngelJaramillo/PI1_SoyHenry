from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import calendar

app = FastAPI()

try:
    df = pd.read_csv('dataset_full.csv')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="File not found: 'dataset_full.csv'")


#El objetivo de esta API es desarrollar 6 funciones para los EndPoints que se consumirán en la API.
#Para las primeras 2 funciones se crea un diccionario para traducir los meses y días al español a sus equivalentes númericos
#También hay que asegurarse que para que las dos primeras funciones se ejecuten correctamente tenemos que tener la columna de release_date con el formato adecuado

df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d', errors='coerce')

meses_espanol = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}

# Diccionario para traducir los días en español a sus equivalentes numéricos
dias_espanol = {
    "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6
}

#1ra función: def cantidad_filmaciones_mes( Mes ): Se ingresa un mes en idioma Español. 
#Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    if mes not in meses_espanol:
        raise HTTPException(status_code=400, detail="Mes inválido. Use el nombre del mes en español, por ejemplo: enero, febrero, etc.")
    
    mes_num = meses_espanol[mes]
    cantidad = df[df['release_date'].dt.month == mes_num].shape[0]
    
    return {"Mes": mes.capitalize(), "Cantidad": cantidad}

#2da función: def cantidad_filmaciones_dia( Dia ): Se ingresa un día en idioma Español. 
#Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dia = dia.lower()
    if dia not in dias_espanol:
        raise HTTPException(status_code=400, detail="Día inválido. Use el nombre del día en español, por ejemplo: lunes, martes, etc.")
    
    dia_num = dias_espanol[dia]
    cantidad = df[df['release_date'].dt.weekday == dia_num].shape[0]
    
    return {"Día": dia.capitalize(), "Cantidad": cantidad}

#3ra función def score_titulo( titulo_de_la_filmación ): 
#Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    movie = df[df['title'].str.contains(titulo, case=False, na=False)]
    if not movie.empty:
        result = movie[['title', 'release_year', 'popularity']].iloc[0].to_dict()
        return result
    else:
        return {"error": "Título no encontrado"}
    
#4ta función: def votos_titulo( titulo_de_la_filmación ): 
#Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. 
# La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    movie = df[df['title'].str.contains(titulo, case=False, na=False)]
    if not movie.empty and movie.iloc[0]['vote_count'] >= 2000:
        result = movie[['title', 'release_year', 'vote_count', 'vote_average']].iloc[0].to_dict()
        return result
    elif not movie.empty:
        return {"error": "La película no tiene suficientes votos"}
    else:
        return {"error": "Título no encontrado"}


#5ta función: def get_actor( nombre_actor ): Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
#Además, la cantidad de películas que en las que ha participado y el promedio de retorno. La definición no deberá considerar directores.
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    actor_movies = df[df['act_name'].str.contains(nombre_actor, case=False, na=False)]
    if not actor_movies.empty:
        total_movies = actor_movies.shape[0]
        total_return = actor_movies['return'].sum()
        avg_return = actor_movies['return'].mean()
        return {
            "actor": nombre_actor,
            "total_movies": total_movies,
            "total_return": total_return,
            "avg_return": avg_return
        }
    else:
        return {"error": "Actor no encontrado"}

#6ta función: def get_director( nombre_director ): Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
# Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    director_movies = df[(df['crew_name'].str.contains(nombre_director, case=False, na=False)) & (df['job'] == 'Director')]
    if not director_movies.empty:
        movies_list = director_movies[['title', 'release_date', 'return', 'budget', 'revenue']].to_dict(orient='records')
        total_return = director_movies['return'].sum()
        return {
            "director": nombre_director,
            "total_return": total_return,
            "movies": movies_list
        }
    else:
        return {"error": "Director no encontrado"}





#Por último el modelo de machine learning de este MVP se basa en un sistema de recomendación
#Este sistema te recomienda 5 peliculas de similar score
#def recomendacion( titulo ): Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.
#Importamos las librerias primero
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException

#Se normalizan los scores
df['score_normalized'] = (df['vote_average'] - df['vote_average'].mean()) / df['vote_average'].std()

#Se crea la función de recomendación
@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    try:
        idx = df[df['title'].str.lower() == titulo.lower()].index[0]
    except IndexError:
        return {"error": "Película no encontrada"}
    
    # Se obtiene el score normalizado de la película dada
    input_score = df.loc[idx, 'score_normalized']

    # Se calcula la diferencia de score
    score_differences = np.abs(df['score_normalized'] - input_score)

    # Y por último se obtienen las 5 películas con scores más similares (excluyendo la película dada)
    similar_indices = score_differences.argsort()[1:6]
    recommendations = df.iloc[similar_indices]['title'].tolist()

    return {"recomendaciones": recommendations}
