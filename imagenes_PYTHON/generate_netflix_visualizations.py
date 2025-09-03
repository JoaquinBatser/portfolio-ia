#!/usr/bin/env python3
"""
Netflix Dataset EDA - Image Generator
This script generates all the visualization images for the Netflix EDA documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure images directory exists
os.makedirs('../public/images/netflix', exist_ok=True)

# Set the style
plt.style.use("classic")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("Loading Netflix dataset...")
# Load the dataset
url = "https://raw.githubusercontent.com/swapnilg4u/Netflix-Data-Analysis/refs/heads/master/netflix_titles.csv"
df = pd.read_csv(url)
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# 1. Generate Missing Data Analysis Visualizations
print("Generating missing data visualizations...")
missing_data = df.isna().sum().sort_values(ascending=False)
missing_percent = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)

# Create visualizations
plt.figure(figsize=(12, 6))

# Subplot 1: Barplot of missing data percentages
plt.subplot(1, 2, 1)
sns.barplot(x=missing_percent[missing_percent > 0].values,
            y=missing_percent[missing_percent > 0].index,
            palette='Reds_r')
plt.title('Porcentaje de Datos Faltantes por Columna')
plt.xlabel('Porcentaje (%)')

# Subplot 2: Heatmap of missing data
plt.subplot(1, 2, 2)
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Patrón de Datos Faltantes')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../public/images/netflix/missing_data.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a separate heatmap with better visibility
plt.figure(figsize=(10, 6))
sns.heatmap(df[missing_data[missing_data > 0].index].isnull(), 
            cmap='viridis', cbar=True)
plt.title('Heatmap de Datos Faltantes')
plt.xticks(rotation=45)
plt.savefig('../public/images/netflix/missing_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Generate Content Types Visualizations
print("Generating content type visualizations...")
type_counts = df['type'].value_counts()
type_percent = df['type'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Countplot 
sns.countplot(data=df, x='type', ax=axes[0], palette='Set2')
axes[0].set_title('Distribución: Movies vs TV Shows')
axes[0].set_ylabel('Cantidad')

# Pie chart
axes[1].pie(type_counts.values, labels=type_counts.index,
           autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
axes[1].set_title('Proporción Movies vs TV Shows')

plt.tight_layout()
plt.savefig('../public/images/netflix/content_types.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Generate Temporal Analysis Visualization
print("Generating temporal analysis visualization...")
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
yearly_releases = df['release_year'].value_counts().sort_index()
recent_years = yearly_releases[yearly_releases.index >= 2000]

plt.figure(figsize=(12, 6))
plt.plot(recent_years.index, recent_years.values,
         marker='o', linewidth=2, markersize=4, color='darkblue')
plt.fill_between(recent_years.index, recent_years.values, alpha=0.3, color='skyblue')
plt.title('Cantidad de Contenido por Año (2000-2021)')
plt.xlabel('Año')
plt.ylabel('Cantidad de Títulos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../public/images/netflix/temporal_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Generate Geographic Distribution Visualization
print("Generating geographic analysis visualization...")
df_countries = df.dropna(subset=['country']).copy()
countries_expanded = df_countries['country'].str.split(', ').explode()
country_counts = countries_expanded.value_counts().head(20)
top_15_countries = country_counts.head(15)

plt.figure(figsize=(10, 8))
sns.barplot(y=top_15_countries.index, x=top_15_countries.values,
           palette='viridis')
plt.title('Top 15 Países con Más Contenido')
plt.xlabel('Cantidad de Títulos')
plt.tight_layout()
plt.savefig('../public/images/netflix/countries_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Generate Ratings Analysis Visualization
print("Generating ratings analysis visualization...")
rating_counts = df['rating'].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Basic countplot
sns.countplot(data=df, x='rating', order=rating_counts.index, 
             ax=axes[0], palette='Set1')
axes[0].set_title('Distribución de Ratings')
axes[0].tick_params(axis='x', rotation=45)

# Countplot by type
sns.countplot(data=df, x='rating', hue='type',
           order=rating_counts.index, ax=axes[1])
axes[1].set_title('Ratings por Tipo de Contenido')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Tipo')

plt.tight_layout()
plt.savefig('../public/images/netflix/ratings_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Generate Genres Analysis Visualization
print("Generating genres analysis visualization...")
genres_expanded = df.dropna(subset=['listed_in'])['listed_in'].str.split(', ').explode()
top_genres = genres_expanded.value_counts().head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Barplot of genres
sns.barplot(y=top_genres.head(10).index, x=top_genres.head(10).values,
           ax=axes[0], palette='Set2')
axes[0].set_title('Top 10 Géneros')
axes[0].set_xlabel('Cantidad')

# Bubble chart for genres
axes[1].scatter(range(len(top_genres)), top_genres.values,
                s=top_genres.values*2, alpha=0.6, 
                c=range(len(top_genres)), cmap='viridis')
for i, (genre, count) in enumerate(top_genres.items()):
    axes[1].annotate(genre, (i, count), ha='center', va='center', fontsize=8)
axes[1].set_title('Bubble Chart - Géneros Populares')
axes[1].set_xticks([])
axes[1].set_xlabel('')
axes[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.savefig('../public/images/netflix/genres_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations have been generated successfully!")
print("Images saved in: ../public/images/netflix/")
