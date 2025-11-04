# Python Test - Extras para Prácticas 8, 9, 10, 11

## 📋 Instalación

### Opción 1: Instalación completa (recomendada)
```bash
pip install -r requirements.txt
```

### Opción 2: Instalación básica (si hay problemas)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Opción 3: Instalación paso a paso
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
# Opcionales (para prácticas específicas):
pip install category-encoders  # Para práctica 9
pip install umap-learn        # Para práctica 10 (opcional)
```

## 📁 Estructura

- `extra_practica8.py`: Extra para Práctica 8 - Feature Engineering con Boston Housing
- `extra_practica9.py`: Extra para Práctica 9 - Comparación CatBoost Encoding vs Target Encoding
- `extra_practica10.py`: Extra para Práctica 10 - Reducción Dimensional No-Lineal (t-SNE, UMAP)
- `extra_practica11.py`: Extra para Práctica 11 - Análisis Temporal con Fourier y Seasonal Decomposition
- `requirements.txt`: Lista de dependencias
- `outputs/`: Carpeta donde se guardan gráficos y resultados

## 🚀 Uso

Ejecuta cada script con:

```bash
python extra_practica8.py
python extra_practica9.py
python extra_practica10.py
python extra_practica11.py
```

## 📊 Salidas

Cada script generará:
- **Gráficos PNG** en `outputs/` (alta resolución, 300 DPI)
- **Archivos de texto** con resultados detallados en `outputs/`
- **Métricas y análisis** completos en consola

## 📝 Descripción de Extras

### Práctica 8: Feature Engineering con Dataset Alternativo
- **Dataset**: Boston Housing (alternativo a Ames Housing)
- **Objetivo**: Validar generalización de técnicas de feature engineering
- **Técnicas**: Ratios, interacciones, transformaciones matemáticas
- **Salidas**: Comparación de importancia, comparación de modelos

### Práctica 9: CatBoost Encoding vs Target Encoding
- **Dataset**: Adult Income (UCI)
- **Objetivo**: Comparar encoding específico para boosting vs genérico
- **Técnicas**: CatBoost Encoding, Target Encoding, evaluación con RF y GB
- **Salidas**: Comparación de accuracy y AUC

### Práctica 10: Reducción Dimensional No-Lineal
- **Dataset**: California Housing / Sintético con estructura no-lineal
- **Objetivo**: Comparar PCA (lineal) vs t-SNE/UMAP (no-lineal)
- **Técnicas**: PCA, t-SNE, UMAP, evaluación en modelos
- **Salidas**: Visualizaciones 2D, comparación de performance

### Práctica 11: Análisis Temporal con Fourier
- **Dataset**: Datos temporales sintéticos con patrones periódicos
- **Objetivo**: Capturar patrones periódicos complejos con análisis espectral
- **Técnicas**: FFT, Seasonal Decomposition, Time Series Cross-Validation
- **Salidas**: Power spectrum, descomposición temporal, comparación de features

## ⚠️ Notas

- Los scripts descargarán datasets automáticamente desde URLs públicas si no están disponibles localmente
- Si falla la descarga, los scripts crearán datasets sintéticos similares
- Algunos scripts requieren librerías opcionales (UMAP, category-encoders) pero funcionan sin ellas
- Todos los scripts usan `random_state=42` para reproducibilidad
- Los resultados se guardan en `outputs/` con nombres descriptivos

## 🐛 Troubleshooting

**Error: "Module not found"**
- Instala las dependencias: `pip install -r requirements.txt`

**Error: "UMAP not available"**
- Es opcional para práctica 10. El script funciona sin UMAP.

**Error: "t-SNE muy lento"**
- Para práctica 10, t-SNE usa solo una muestra de 500 datos por defecto.

**Python 3.14 compatibility issues**
- Algunas librerías pueden no estar disponibles. Usa las básicas primero.

