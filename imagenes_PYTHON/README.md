# Netflix EDA - Generador de Visualizaciones

Este directorio contiene scripts para generar las visualizaciones usadas en la documentación del EDA de Netflix.

## Contenido

- `generate_netflix_visualizations.py`: Script que genera todas las visualizaciones necesarias para la documentación.

## Instrucciones de uso

1. Asegúrate de tener instaladas las siguientes dependencias:

   - pandas
   - numpy
   - matplotlib
   - seaborn

2. Ejecuta el script desde esta carpeta:

   ```
   python generate_netflix_visualizations.py
   ```

3. Las imágenes se generarán en la carpeta `../public/images/netflix/`

## Visualizaciones generadas

- `missing_data.png`: Análisis de datos faltantes
- `missing_heatmap.png`: Heatmap de datos faltantes
- `content_types.png`: Distribución de tipos de contenido
- `temporal_trend.png`: Tendencia temporal de contenidos
- `countries_distribution.png`: Distribución geográfica
- `ratings_analysis.png`: Análisis de ratings
- `genres_analysis.png`: Análisis de géneros

## Nota

Asegúrate de que existe la carpeta `../public/images/netflix/` antes de ejecutar el script, o ejecutalo con permisos suficientes para crear directorios.
