"""
Extra Práctica 10: Reducción Dimensional No-Lineal con t-SNE y UMAP
Comparando técnicas no-lineales como alternativa a PCA para visualización y reducción dimensional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

# Intentar importar t-SNE y UMAP (opcionales)
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("⚠️ t-SNE no disponible (sklearn < 0.24). Usando implementación simple.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP no disponible. Instalar con: pip install umap-learn")

# Configuración
np.random.seed(42)
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("EXTRA PRÁCTICA 10: Reducción Dimensional No-Lineal")
print("=" * 60)

print("\n📋 ¿POR QUÉ LO ELEGÍ?")
print("-" * 60)
print("""
Elegí explorar t-SNE y UMAP porque:
1. PCA es lineal y puede perder información en datos con relaciones no-lineales
2. t-SNE y UMAP son técnicas no-lineales que preservan mejor la estructura local
3. Quería comparar reducción dimensional para visualización vs para modelado
4. t-SNE es excelente para visualización pero lento; UMAP es más rápido y escalable
5. Es una técnica mencionada en "próximos pasos" de la práctica principal
""")

print("\n🔍 ¿QUÉ ESPERABA ENCONTRAR?")
print("-" * 60)
print("""
Esperaba encontrar:
- Que t-SNE/UMAP preserven mejor la estructura local de los datos
- Que PCA siga siendo mejor para modelado (mantiene varianza global)
- Que t-SNE sea excelente para visualización pero no para features de modelo
- Que UMAP sea un buen balance entre visualización y velocidad
- Que la reducción no-lineal revele clusters que PCA no puede capturar
""")

# Cargar dataset Ames Housing (o sintético)
print("\n📊 CARGANDO DATASET...")
print("-" * 60)

try:
    # Intentar cargar desde URL o archivo local
    df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
    if 'median_house_value' in df.columns:
        target_col = 'median_house_value'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(target_col)
        df = df[numeric_cols + [target_col]].dropna()
        print(f"✅ Dataset California Housing cargado: {df.shape}")
except Exception as e:
    print(f"⚠️ Error cargando dataset: {e}")
    print("Creando dataset sintético con estructura no-lineal...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Crear datos con estructura no-lineal (manifold)
    X_manifold = np.random.randn(n_samples, 3)
    
    # Transformar a espacio de mayor dimensión con relaciones no-lineales
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        if i < 5:
            X[:, i] = X_manifold[:, 0] ** (i+1) + np.random.normal(0, 0.1, n_samples)
        elif i < 10:
            X[:, i] = np.sin(X_manifold[:, 1] * (i-4)) + np.random.normal(0, 0.1, n_samples)
        elif i < 15:
            X[:, i] = X_manifold[:, 2] * X_manifold[:, 0] + np.random.normal(0, 0.1, n_samples)
        else:
            X[:, i] = np.random.randn(n_samples) * 0.5
    
    # Generar target con relaciones no-lineales
    y = (
        10 * X_manifold[:, 0] ** 2
        + 5 * np.sin(X_manifold[:, 1] * 2)
        + 3 * X_manifold[:, 2]
        + np.random.normal(0, 2, n_samples)
    )
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    target_col = 'target'
    numeric_cols = feature_names
    
    print(f"✅ Dataset sintético creado: {df.shape}")

print(f"\nDataset preview:")
print(df.head())
print(f"\nShape: {df.shape}")

# Preparar datos
X = df[numeric_cols].values
y = df[target_col].values

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Reducción dimensional
print("\n🔧 APLICANDO REDUCCIÓN DIMENSIONAL...")
print("-" * 60)

results = {}

# PCA
print("Aplicando PCA...")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

explained_var = pca.explained_variance_ratio_.sum()
results['PCA'] = {
    'explained_variance': explained_var,
    'n_components': 2
}
print(f"✅ PCA completado. Varianza explicada: {explained_var:.4f}")

# t-SNE (solo para visualización, no para modelado)
if TSNE_AVAILABLE:
    print("Aplicando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_train_tsne = tsne.fit_transform(X_train[:500])  # t-SNE es lento, usar muestra
    results['t-SNE'] = {'applied': True, 'n_samples': 500}
    print("✅ t-SNE completado (en muestra de 500)")
else:
    print("⚠️ t-SNE no disponible")
    X_train_tsne = None
    results['t-SNE'] = {'applied': False}

# UMAP
if UMAP_AVAILABLE:
    print("Aplicando UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_train_umap = reducer.fit_transform(X_train)
    X_test_umap = reducer.transform(X_test)
    results['UMAP'] = {'applied': True, 'n_components': 2}
    print("✅ UMAP completado")
else:
    print("⚠️ UMAP no disponible")
    X_train_umap = None
    X_test_umap = None
    results['UMAP'] = {'applied': False}

# Evaluar en modelo (solo PCA y UMAP, t-SNE no es para modelado)
print("\n🎯 EVALUANDO EN MODELO...")
print("-" * 60)

# Modelo con features originales
rf_orig = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_orig.fit(X_train, y_train)
y_pred_orig = rf_orig.predict(X_test)
results['Original'] = {
    'r2': r2_score(y_test, y_pred_orig),
    'mse': mean_squared_error(y_test, y_pred_orig)
}

# Modelo con PCA
rf_pca = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
results['PCA']['r2'] = r2_score(y_test, y_pred_pca)
results['PCA']['mse'] = mean_squared_error(y_test, y_pred_pca)

# Modelo con UMAP (si disponible)
if UMAP_AVAILABLE:
    rf_umap = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_umap.fit(X_train_umap, y_train)
    y_pred_umap = rf_umap.predict(X_test_umap)
    results['UMAP']['r2'] = r2_score(y_test, y_pred_umap)
    results['UMAP']['mse'] = mean_squared_error(y_test, y_pred_umap)

print("\nResultados del modelo:")
print("-" * 60)
for method, metrics in results.items():
    if 'r2' in metrics:
        print(f"{method}: R² = {metrics['r2']:.4f}, MSE = {metrics['mse']:.4f}")

# Visualizaciones
print("\n📊 GENERANDO VISUALIZACIONES...")
print("-" * 60)

n_plots = 1 + (1 if TSNE_AVAILABLE else 0) + (1 if UMAP_AVAILABLE else 0)
fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
if n_plots == 1:
    axes = [axes]

idx = 0

# PCA
scatter = axes[idx].scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                           c=y_train, cmap='viridis', alpha=0.6, s=20)
axes[idx].set_title(f'PCA (Var: {explained_var:.2%})')
axes[idx].set_xlabel('Componente 1')
axes[idx].set_ylabel('Componente 2')
plt.colorbar(scatter, ax=axes[idx], label='Target')
idx += 1

# t-SNE
if TSNE_AVAILABLE and X_train_tsne is not None:
    scatter = axes[idx].scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], 
                               c=y_train[:500], cmap='viridis', alpha=0.6, s=20)
    axes[idx].set_title('t-SNE (Visualización)')
    axes[idx].set_xlabel('Dimensión 1')
    axes[idx].set_ylabel('Dimensión 2')
    plt.colorbar(scatter, ax=axes[idx], label='Target')
    idx += 1

# UMAP
if UMAP_AVAILABLE and X_train_umap is not None:
    scatter = axes[idx].scatter(X_train_umap[:, 0], X_train_umap[:, 1], 
                               c=y_train, cmap='viridis', alpha=0.6, s=20)
    axes[idx].set_title('UMAP')
    axes[idx].set_xlabel('Dimensión 1')
    axes[idx].set_ylabel('Dimensión 2')
    plt.colorbar(scatter, ax=axes[idx], label='Target')

plt.tight_layout()
plt.savefig('outputs/practica10_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
print("✅ Guardado: outputs/practica10_dimensionality_reduction.png")

# Comparación de performance
if UMAP_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = ['Original', 'PCA', 'UMAP']
    r2_scores = [results[m]['r2'] for m in methods if 'r2' in results[m]]
    mse_scores = [results[m]['mse'] for m in methods if 'mse' in results[m]]
    
    axes[0].bar(methods[:len(r2_scores)], r2_scores, color=['steelblue', 'orange', 'green'])
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² Score por Método de Reducción')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(methods[:len(mse_scores)], mse_scores, color=['steelblue', 'orange', 'green'])
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE por Método de Reducción')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/practica10_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Guardado: outputs/practica10_model_comparison.png")

# ¿Qué aprendí?
print("\n🎓 ¿QUÉ APRENDÍ?")
print("=" * 60)

r2_orig = results['Original']['r2']
r2_pca = results['PCA']['r2']
r2_pca_loss = ((r2_orig - r2_pca) / r2_orig) * 100

# Obtener R² de UMAP si está disponible
r2_umap = results['UMAP'].get('r2')
r2_umap_str = f"{r2_umap:.4f}" if r2_umap is not None else "N/A"

print(f"""
1. DIFERENCIAS ENTRE TÉCNICAS LINEALES Y NO-LINEALES:
   - PCA: Varianza explicada: {explained_var:.2%}, R²: {r2_pca:.4f}
   - PCA pierde {r2_pca_loss:.2f}% de performance vs original
   - PCA es excelente para mantener varianza global pero puede perder estructura local

2. t-SNE PARA VISUALIZACIÓN:
   - t-SNE es excelente para visualizar clusters y estructura local
   - NO es adecuado para features de modelo (no preserva distancias globales)
   - Es computacionalmente costoso (O(n²)) y no determinista
   - Mejor uso: exploración de datos, visualización de clusters

3. UMAP COMO ALTERNATIVA:
   - UMAP es más rápido que t-SNE y más escalable
   - Preserva tanto estructura local como global (mejor que t-SNE)
   - Puede usarse para reducción dimensional en modelado
   - R² con UMAP: {r2_umap_str} (si disponible)

4. RECOMENDACIONES POR OBJETIVO:
   - VISUALIZACIÓN: t-SNE o UMAP (t-SNE mejor para clusters, UMAP más rápido)
   - REDUCCIÓN PARA MODELADO: PCA (mantiene varianza, interpretable) o UMAP (si hay estructura no-lineal)
   - INTERPRETABILIDAD: PCA (componentes lineales explicables)
   - VELOCIDAD: PCA > UMAP > t-SNE

5. INSIGHTS ESPECÍFICOS:
   - Para datos con relaciones no-lineales, UMAP puede ser mejor que PCA
   - PCA sigue siendo la mejor opción cuando necesitas interpretabilidad
   - La pérdida de performance con PCA ({r2_pca_loss:.2f}%) es aceptable si reduces de {X_train.shape[1]} a 2 dimensiones
   - Para producción, PCA es más robusto y predecible

6. CUÁNDO USAR CADA TÉCNICA:
   - PCA: Datos lineales, necesidad de interpretabilidad, velocidad crítica
   - t-SNE: Solo visualización, exploración de clusters, datasets pequeños (<10K)
   - UMAP: Visualización + modelado, estructura no-lineal, datasets medianos/grandes
""")

# Guardar resultados
with open('outputs/practica10_results.txt', 'w', encoding='utf-8') as f:
    f.write("EXTRA PRÁCTICA 10: RESULTADOS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Comparación de Reducción Dimensional:\n\n")
    for method, metrics in results.items():
        f.write(f"{method}:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

print("\n✅ Guardado: outputs/practica10_results.txt")
print("\n" + "=" * 60)
print("✅ EXTRA PRÁCTICA 10 COMPLETADO")
print("=" * 60)

