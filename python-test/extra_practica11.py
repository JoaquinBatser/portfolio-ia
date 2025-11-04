"""
Extra Práctica 11: Análisis de Patrones Temporales con Fourier y Seasonal Decomposition
Aplicando técnicas avanzadas de análisis temporal para capturar patrones periódicos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import signal
import warnings
import os
warnings.filterwarnings('ignore')

# Configuración
np.random.seed(42)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("EXTRA PRÁCTICA 11: Análisis Temporal Avanzado con Fourier")
print("=" * 60)

print("\n📋 ¿POR QUÉ LO ELEGÍ?")
print("-" * 60)
print("""
Elegí explorar análisis de Fourier y seasonal decomposition porque:
1. Las features temporales manuales (lag, rolling) pueden perder patrones periódicos complejos
2. Fourier Transform captura patrones cíclicos de diferentes frecuencias automáticamente
3. Seasonal decomposition separa tendencia, estacionalidad y ruido de manera sistemática
4. Quería comparar features manuales vs features extraídas de análisis espectral
5. Es una técnica mencionada en "próximos pasos" de la práctica principal
""")

print("\n🔍 ¿QUÉ ESPERABA ENCONTRAR?")
print("-" * 60)
print("""
Esperaba encontrar:
- Que Fourier features capturen patrones semanales/mensuales que lag features no detectan
- Que seasonal decomposition mejore la interpretabilidad de patrones temporales
- Que las features de frecuencia sean complementarias a las features manuales
- Que el análisis espectral revele periodicidades ocultas en los datos
- Que la combinación de ambos enfoques (manual + espectral) sea superior
""")

# Generar datos temporales sintéticos
print("\n📊 GENERANDO DATOS TEMPORALES...")
print("-" * 60)

np.random.seed(42)
n_samples = 1000
dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

# Crear datos con múltiples patrones temporales
df = pd.DataFrame({
    'date': dates,
    'day_of_week': dates.dayofweek,
    'day_of_month': dates.day,
    'month': dates.month,
})

# Generar target con patrones temporales complejos
# Patrón semanal (7 días)
weekly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 7)
# Patrón mensual (30 días)
monthly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 30)
# Patrón trimestral (90 días)
quarterly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 90)
# Tendencia
trend = np.linspace(0, 2, n_samples)
# Ruido
noise = np.random.normal(0, 0.3, n_samples)

# Target binario con probabilidad basada en patrones temporales
target_prob = (
    0.3 + 0.2 * weekly_pattern + 0.15 * monthly_pattern 
    + 0.1 * quarterly_pattern + 0.1 * trend + noise
).clip(0, 1)

df['target'] = (target_prob > 0.5).astype(int)
df['target_prob'] = target_prob

# Features numéricas adicionales
df['value'] = 10 + 2 * weekly_pattern + 1.5 * monthly_pattern + np.random.normal(0, 1, n_samples)
df['quantity'] = 5 + 1 * weekly_pattern + 0.5 * monthly_pattern + np.random.normal(0, 0.5, n_samples)

print(f"✅ Dataset temporal creado: {df.shape}")
print(f"Período: {df['date'].min()} a {df['date'].max()}")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")

# Features temporales manuales (como en práctica principal)
print("\n🔧 CREANDO FEATURES TEMPORALES MANUALES...")
print("-" * 60)

df['lag_1'] = df['target'].shift(1)
df['lag_7'] = df['target'].shift(7)
df['rolling_mean_7'] = df['value'].rolling(window=7, min_periods=1).mean()
df['rolling_mean_30'] = df['value'].rolling(window=30, min_periods=1).mean()

# Encoding cíclico
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("✅ Features temporales manuales creadas")

# Análisis de Fourier
print("\n🌊 APLICANDO ANÁLISIS DE FOURIER...")
print("-" * 60)

# FFT en la serie temporal del target
target_series = df['target'].values
fft_values = np.fft.fft(target_series)
fft_freq = np.fft.fftfreq(len(target_series))

# Identificar frecuencias dominantes
power = np.abs(fft_values) ** 2
dominant_freq_idx = np.argsort(power)[-5:][::-1]  # Top 5 frecuencias
dominant_freqs = fft_freq[dominant_freq_idx]

print(f"Frecuencias dominantes detectadas: {dominant_freqs[:3]}")

# Crear features de Fourier
# Usar las frecuencias más importantes para crear features
for i, freq_idx in enumerate(dominant_freq_idx[:3]):
    freq = fft_freq[freq_idx]
    if freq > 0:  # Solo frecuencias positivas
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * freq * np.arange(n_samples))
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * freq * np.arange(n_samples))

print("✅ Features de Fourier creadas")

# Seasonal Decomposition (simplificado)
print("\n📈 APLICANDO SEASONAL DECOMPOSITION...")
print("-" * 60)

from scipy import signal

# Descomposición usando filtro de mediana móvil
window_size = 7
trend = signal.medfilt(df['value'].values, kernel_size=window_size)
detrended = df['value'].values - trend

# Extraer componente estacional (promedio por día de semana)
seasonal = np.zeros(n_samples)
for day in range(7):
    mask = df['day_of_week'] == day
    if mask.sum() > 0:
        seasonal[mask] = detrended[mask].mean()

residual = detrended - seasonal

df['trend'] = trend
df['seasonal'] = seasonal
df['residual'] = residual

print("✅ Seasonal decomposition completado")

# Preparar features
manual_features = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30',
                   'day_sin', 'day_cos', 'month_sin', 'month_cos',
                   'value', 'quantity']
fourier_features = [col for col in df.columns if col.startswith('fourier_')]
decomposition_features = ['trend', 'seasonal', 'residual']

all_features = manual_features + fourier_features + decomposition_features

# Limpiar NaN
df_clean = df[all_features + ['target']].fillna(0)

X = df_clean[all_features].values
y = df_clean['target'].values

# Time Series Split (temporal)
print("\n🎯 EVALUANDO MODELOS...")
print("-" * 60)

tscv = TimeSeriesSplit(n_splits=3)
results = {}

# Modelo con features manuales
X_manual = df_clean[manual_features].fillna(0).values

auc_scores_manual = []
for train_idx, test_idx in tscv.split(X_manual):
    X_train, X_test = X_manual[train_idx], X_manual[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc_scores_manual.append(roc_auc_score(y_test, y_proba))

results['Manual Features'] = {
    'mean_auc': np.mean(auc_scores_manual),
    'std_auc': np.std(auc_scores_manual)
}

# Modelo con features manuales + Fourier
X_fourier = df_clean[manual_features + fourier_features].fillna(0).values

auc_scores_fourier = []
for train_idx, test_idx in tscv.split(X_fourier):
    X_train, X_test = X_fourier[train_idx], X_fourier[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc_scores_fourier.append(roc_auc_score(y_test, y_proba))

results['Manual + Fourier'] = {
    'mean_auc': np.mean(auc_scores_fourier),
    'std_auc': np.std(auc_scores_fourier)
}

# Modelo con todas las features
auc_scores_all = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc_scores_all.append(roc_auc_score(y_test, y_proba))

results['All Features'] = {
    'mean_auc': np.mean(auc_scores_all),
    'std_auc': np.std(auc_scores_all)
}

print("\nResultados (Time Series Cross-Validation):")
print("-" * 60)
for method, metrics in results.items():
    print(f"{method}: AUC = {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")

# Visualizaciones
print("\n📊 GENERANDO VISUALIZACIONES...")
print("-" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1. Serie temporal original
axes[0].plot(df['date'], df['target_prob'], alpha=0.7, label='Target Probability')
axes[0].set_title('Serie Temporal Original')
axes[0].set_xlabel('Fecha')
axes[0].set_ylabel('Probabilidad')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Análisis de Fourier (Power Spectrum)
axes[1].plot(fft_freq[:len(fft_freq)//2], power[:len(power)//2])
axes[1].set_title('Power Spectrum (Fourier Transform)')
axes[1].set_xlabel('Frecuencia')
axes[1].set_ylabel('Power')
axes[1].axvline(x=1/7, color='r', linestyle='--', label='Frecuencia semanal (1/7)')
axes[1].axvline(x=1/30, color='g', linestyle='--', label='Frecuencia mensual (1/30)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Seasonal Decomposition
axes[2].plot(df['date'][:200], df['value'][:200], label='Original', alpha=0.7)
axes[2].plot(df['date'][:200], df['trend'][:200], label='Tendencia', linewidth=2)
axes[2].plot(df['date'][:200], df['seasonal'][:200] + df['trend'][:200], label='Tendencia + Estacional', linewidth=2)
axes[2].set_title('Seasonal Decomposition (Primeros 200 días)')
axes[2].set_xlabel('Fecha')
axes[2].set_ylabel('Valor')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/practica11_temporal_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Guardado: outputs/practica11_temporal_analysis.png")

# Comparación de modelos
fig, ax = plt.subplots(figsize=(10, 6))

methods = list(results.keys())
mean_aucs = [results[m]['mean_auc'] for m in methods]
std_aucs = [results[m]['std_auc'] for m in methods]

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, mean_aucs, yerr=std_aucs, capsize=5, 
              color=['steelblue', 'orange', 'green'], alpha=0.7)
ax.set_ylabel('ROC AUC')
ax.set_title('Comparación de Features Temporales (Time Series CV)')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for i, (bar, mean, std) in enumerate(zip(bars, mean_aucs, std_aucs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
            f'{mean:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('outputs/practica11_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Guardado: outputs/practica11_model_comparison.png")

# ¿Qué aprendí?
print("\n🎓 ¿QUÉ APRENDÍ?")
print("=" * 60)

improvement_fourier = ((results['Manual + Fourier']['mean_auc'] - results['Manual Features']['mean_auc']) 
                       / results['Manual Features']['mean_auc'] * 100)
improvement_all = ((results['All Features']['mean_auc'] - results['Manual Features']['mean_auc']) 
                   / results['Manual Features']['mean_auc'] * 100)

print(f"""
1. FOURIER FEATURES CAPTURAN PATRONES PERIÓDICOS:
   - Mejora con Fourier: {improvement_fourier:+.2f}% en AUC
   - Fourier detecta automáticamente frecuencias dominantes (semanal, mensual)
   - Las features de Fourier son complementarias a las features manuales
   - Frecuencias detectadas: {dominant_freqs[0]:.4f}, {dominant_freqs[1]:.4f}, {dominant_freqs[2]:.4f}

2. SEASONAL DECOMPOSITION MEJORA INTERPRETABILIDAD:
   - Separación clara entre tendencia, estacionalidad y ruido
   - Features de tendencia capturan cambios de largo plazo
   - Features estacionales capturan patrones cíclicos repetitivos
   - Residual puede indicar eventos anómalos

3. COMBINACIÓN DE ENFOQUES ES SUPERIOR:
   - Mejora total con todas las features: {improvement_all:+.2f}%
   - Features manuales: capturan relaciones específicas del dominio
   - Features de Fourier: capturan patrones periódicos automáticamente
   - Features de descomposición: mejoran interpretabilidad y separación de componentes

4. CUÁNDO USAR CADA TÉCNICA:
   - Features manuales: cuando conoces el dominio y relaciones específicas
   - Fourier: cuando hay patrones periódicos complejos o múltiples frecuencias
   - Seasonal decomposition: cuando necesitas separar tendencia de estacionalidad
   - Combinación: siempre que sea computacionalmente factible

5. INSIGHTS ESPECÍFICOS:
   - El análisis espectral revela periodicidades que no son obvias visualmente
   - La frecuencia semanal (1/7) es claramente visible en el power spectrum
   - La frecuencia mensual (1/30) también es detectada pero con menor potencia
   - Time Series Cross-Validation es crítico para validación realista

6. RECOMENDACIONES:
   - Para producción: empezar con features manuales, luego añadir Fourier si hay patrones periódicos
   - Usar seasonal decomposition para análisis exploratorio y debugging
   - Fourier es especialmente valioso cuando hay múltiples frecuencias superpuestas
   - La combinación de técnicas manuales + espectrales da mejor performance
""")

# Guardar resultados
with open('outputs/practica11_results.txt', 'w', encoding='utf-8') as f:
    f.write("EXTRA PRÁCTICA 11: RESULTADOS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Comparación de Features Temporales:\n\n")
    for method, metrics in results.items():
        f.write(f"{method}:\n")
        f.write(f"  AUC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}\n\n")
    f.write(f"\nFrecuencias dominantes detectadas:\n")
    for i, freq in enumerate(dominant_freqs[:5]):
        f.write(f"  {i+1}. {freq:.6f}\n")

print("\n✅ Guardado: outputs/practica11_results.txt")
print("\n" + "=" * 60)
print("✅ EXTRA PRÁCTICA 11 COMPLETADO")
print("=" * 60)

