#PYTHON FILE WITH ALL MY FUNCTIONS USEFUL FOR EDA 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os
import scipy.stats as ss 
import time
import gc
import psutil

#este archivo no sigue las normativas de explicaciones de funciones y ha sido realizado en inglés/francés/espanol,
#no sabiendo las reglas lo déjé asi, tratando explicar en castellano en mis notebooks lo que hacen mis funciones de manera clara: 





# HANDLING MISSING VALUES

def check_missing_values(df):
    #Displays the percentage of missing values for each column
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]
    print("% of NAN per column:\n", missing)
    return missing


def check_missing_per_row(df):
    # Displays the percentage of missing values for each row, sorted in descending order, along with the target column
    missing_percentage_per_row = df.isnull().mean(axis=1) * 100
    missing_row_df = pd.DataFrame(missing_percentage_per_row, columns=['missing_percentage'])
    missing_row_df['TARGET'] = df['TARGET']
    missing_row_df_sorted = missing_row_df.sort_values(by='missing_percentage', ascending=False)
    return missing_row_df_sorted


def plot_missing_values(df):
    # Plots a bar chart of the percentage of missing values for each column
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values.")
    else:
        missing.sort_values(inplace=True)
        missing.plot.bar(figsize=(10, 6), color='darkblue')
        plt.title("% of NAN per column")
        plt.xlabel("Column")
        plt.ylabel("% of NAN")
        plt.show()


def fill_missing_with_mean(df):
    # Fills missing values in numerical columns with the column mean
    for column in df.select_dtypes(include=[np.number]):
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
    print("Missing values replaced with the mean.")
    return df




















# DATA SUMMARY & DESCRIPTION

def check_duplicate_columns(df):
    duplicate_columns = {}
    
    # Compare each column with all others
    for col in df.columns:
        duplicates = [other_col for other_col in df.columns if col != other_col and df[col].equals(df[other_col])]
        if duplicates:
            duplicate_columns[col] = duplicates
    
    return duplicate_columns

def describe_data(df):
    # Returns statistical summary for numerical and categorical columns
    numeric_desc = df.describe()
    categorical_desc = df.describe(include=['object', 'category'])
    print("Statistical summary for numerical columns:\n", numeric_desc)
    print("\nStatistical summary for categorical columns:\n", categorical_desc)
    return numeric_desc, categorical_desc


def unique_values_per_category(df):
    # Displays the number of unique values for each categorical column
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    unique_counts = {col: df[col].nunique() for col in cat_cols}
    print("Number of unique values per categorical column:\n", unique_counts)
    return unique_counts






















# OUTLIER DETECTION

def detect_outliers_zscore(df, threshold=3):
    # Detects outliers in numerical columns using the Z-score method
    z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
    outliers = (z_scores > threshold).any(axis=1)
    print(f"Number of outliers detected (threshold={threshold}):", outliers.sum())
    return df[outliers]


def plot_boxplots(df):
    # Displays boxplots for numerical columns to detect outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols].plot(kind='box', subplots=True, layout=(len(numeric_cols) // 3 + 1, 3), figsize=(15, 10))
    plt.suptitle("Boxplot of numerical variables")
    plt.show()



#VARIABLES CONTINUAS Y CATEGORICAS 

#CATEGORICAL AND NUMERICAL HANDLING

def dame_variables_categoricas(dataset=None):
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    categorical_vars = []  # Lista para las variables categóricas
    continuous_vars = []  # Lista para las variables numéricas

    for i in dataset.columns:
        # Vérifie le type de données
        if (dataset[i].dtype != 'float64') & (dataset[i].dtype != 'int64'):
            # Cas des variables non numériques
            unicos = int(len(np.unique(dataset[i].dropna())))  # Nombre de valeurs uniques
            if unicos < 100:
                categorical_vars.append(i)  # Ajouter à la liste des variables catégoriques
            else:
                continuous_vars.append(i)  # Ajouter aux autres variables
        else:
            # Cas des variables numériques (float ou int)
            unicos = int(len(np.unique(dataset[i].dropna())))  # Nombre de valeurs uniques
            if unicos < 20:  # Si moins de 20 valeurs uniques
                categorical_vars.append(i)  # Ajouter à la liste des variables catégoriques
            else:
                continuous_vars.append(i)  # Ajouter aux autres variables

    return categorical_vars, continuous_vars



def convert_object_to_categorical(df, categorical_cols):
    # Converts columns of type object to categorical
    for col in categorical_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    return df


def map_category_columns(df):
    # Maps categorical values 'Y' and 'N' to 'YES' and 'NO'
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].str.upper().replace({'Y': 'YES', 'N': 'NO'})
    return df


def check_binary_like_columns(df, numeric_columns):
    # Checks for numerical columns with exactly two unique values (binary-like)
    binary_like_columns = []
    for col in numeric_columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) == 2:
            binary_like_columns.append(col)
    return binary_like_columns



# VISUALIZATION FUNCTIONS



#CATEGORICAS 


def analyze_variable_characteristics(df, col_name):
    
    n_unique = df[col_name].nunique()
    n_total = len(df)
    
    # Caractéristiques de la distribution
    if df[col_name].dtype in ['int64', 'float64']:
        skewness = df[col_name].skew()
        is_numeric = True
    else:
        skewness = 0
        is_numeric = False
    
    return {
        'is_numeric': is_numeric,
        'n_unique': n_unique,
        'n_unique_ratio': n_unique / n_total if n_total > 0 else 0,
        'skewness': skewness
    }


def plot_smart_visualization(df, col_name, target, figsize=(15, 6)):

    plt.style.use('seaborn')
    characteristics = analyze_variable_characteristics(df, col_name)
    
    # Gestion des valeurs manquantes
    df_copy = df.copy()
    missing_count = df[col_name].isnull().sum()
    total_count = len(df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Premier graphique : Distribution
    if characteristics['is_numeric']:
        if characteristics['n_unique_ratio'] < 0.05:  # Peu de valeurs uniques par rapport à la taille
            # Utiliser un barplot pour les variables numériques discrètes
            value_counts = df_copy[col_name].value_counts().sort_index()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax1, color='darkblue', alpha=0.6)
            
            # Ajouter les pourcentages
            for i, v in enumerate(value_counts.values):
                ax1.text(i, v, f'{(v/total_count)*100:.1f}%', ha='center', va='bottom')
        else:
            # Histogramme avec KDE pour les variables continues
            sns.histplot(data=df_copy, x=col_name, ax=ax1, kde=True, color='darkblue', alpha=0.5)
            
            # Ajouter les statistiques descriptives
            stats = df[col_name].describe()
            stats_text = f'Mean: {stats["mean"]:.2f}\nStd: {stats["std"]:.2f}\nSkewness: {characteristics["skewness"]:.2f}'
            
            ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
                     horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Pour les variables catégorielles
        value_counts = df_copy[col_name].fillna('Missing').value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax1, palette='viridis')
        
        for i, v in enumerate(value_counts.values):
            ax1.text(i, v, f'{(v/total_count)*100:.1f}%', ha='center', va='bottom')
    
    ax1.set_title(f'Distribution of {col_name}\n(Missing: {missing_count:,} values, {(missing_count/total_count)*100:.1f}%)',
                  pad=20)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    
    # Rotation des labels si nécessaire
    if characteristics['n_unique'] > 5 or df_copy[col_name].astype(str).map(len).max() > 10:
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Deuxième graphique : Relation avec la variable cible
    if characteristics['is_numeric']:
        if characteristics['n_unique_ratio'] < 0.05:  # Variables numériques discrètes
            # Utiliser un barplot empilé
            cross_tab = pd.crosstab(df_copy[col_name], df_copy[target], normalize='index') * 100
            cross_tab.plot(kind='bar', stacked=True, ax=ax2, colormap='coolwarm')
            ax2.set_ylabel('Percentage (%)')
        else:
            # Boxplot pour les variables continues
            sns.boxplot(data=df_copy, x=target, y=col_name, ax=ax2, palette='viridis')
            ax2.set_ylabel(col_name)
            
            # Ajouter un violinplot en transparence pour plus d'information
            sns.violinplot(data=df_copy, x=target, y=col_name, ax=ax2, alpha=0.2, color='gray')
    else:
        # Pour les variables catégorielles
        cross_tab = pd.crosstab(df_copy[col_name].fillna('Missing'), df_copy[target], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=True, ax=ax2, colormap='coolwarm')
        ax2.set_ylabel('Percentage (%)')
    
    ax2.set_title(f'Relationship with {target}', pad=20)
    ax2.set_xlabel(target)
    
    # Rotation des labels si nécessaire
    if characteristics['n_unique'] > 5 or df_copy[col_name].astype(str).map(len).max() > 10:
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Ajout de la légende si nécessaire
    if not characteristics['is_numeric'] or characteristics['n_unique_ratio'] < 0.05:
        ax2.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    # Modification : Sauvegarder dans '../images/02_notebook_images'
    output_dir = '../images/02_notebook_images'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{col_name}_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_variables(df, variables, target):
    
    for var in variables:
        print(f"\nAnalyzing {var}")
        try:
            plot_smart_visualization(df, var, target)
        except Exception as e:
            print(f"Error plotting {var}: {str(e)}")






#CONTINUAS 




def set_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def clear_memory():
    plt.close('all')
    gc.collect()


def optimize_figure_size(n_categories, label_length):
    if n_categories > 10 or label_length > 15:
        return (15, 6)
    return (12, 5)


def plot_feature(df, col_name, isContinuous, target, max_categories=15, sample_size=50000):
    
    # Muestreo si el dataset es grande
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Calcular métricas para el tamaño
    n_unique = len(df[col_name].unique())
    max_label_len = df[col_name].astype(str).str.len().max()
    figsize = optimize_figure_size(n_unique, max_label_len)

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Paleta de colores personalizada
    palette = sns.color_palette(['#2ecc71', '#e74c3c', '#3498db'])

    try:
        # Calcular información de nulos
        nulls = df[col_name].isnull().sum()
        null_pct = (nulls / len(df)) * 100

        if isContinuous:
            # Limpiar outliers
            q1, q3 = df[col_name].quantile([0.01, 0.99])
            df_clean = df[(df[col_name] >= q1) & (df[col_name] <= q3)]

            # Gráfico de distribución
            sns.histplot(
                data=df_clean,
                x=col_name,
                ax=ax1,
                color=palette[0],
                kde=True,
                bins=30
            )

            # Añadir líneas de media y mediana
            mean_val = df_clean[col_name].mean()
            median_val = df_clean[col_name].median()
            ax1.axvline(mean_val, color=palette[1], linestyle='--', label=f'Media: {mean_val:.2f}')
            ax1.axvline(median_val, color=palette[2], linestyle='--', label=f'Mediana: {median_val:.2f}')
            ax1.legend(fontsize=8)

            # Boxplot
            sns.boxplot(
                x=target,
                y=col_name,
                data=df_clean,
                ax=ax2,
                palette=[palette[0], palette[1]]
            )

        else:
            # No procesamos variables categóricas en este caso
            return

        # Configurar títulos y etiquetas
        ax1.set_title(f'Distribución de {col_name}\nNulos: {nulls:,} ({null_pct:.1f}%)')
        ax2.set_title(f'Relación con {target}')

        # Ajustar etiquetas y grid
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(col_name, fontsize=9)

        # Ajustar espacio
        plt.tight_layout()

        # --------- MODIFICATION DE LA ROUTE ---------
        # Directorio donde se guardarán los gráficos
        output_dir = os.path.join('..', 'images', '02_notebook_images')

        # Crear el directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Guardar la figura con el nombre de la variable
        fig_path = os.path.join(output_dir, f"{col_name}.png")
        plt.savefig(fig_path)

        # Mostrar la figura en el notebook
        plt.show()

        # Cerrar la figura para liberar memoria
        plt.close(fig)

        print(f"Gráfico de '{col_name}' guardado exitosamente en '{output_dir}'.")
        # ------------------------------------------------------

    except Exception as e:
        plt.close(fig)
        print(f"Error en {col_name}: {str(e)}")
        return None


def plot_all_features(df, continuous_vars, target_col='TARGET', batch_size=3, memory_threshold=85):
    
    # Configurar estilo
    set_plot_style()
    
    # Preparar columnas
    columns = continuous_vars  # Usamos las variables continuas pasadas como argumento
    total_cols = len(columns)
    
    print(f"Iniciando análisis de {total_cols} características continuas")
    try:
        for i in range(0, total_cols, batch_size):
            # Verificar memoria
            if psutil.virtual_memory().percent > memory_threshold:
                clear_memory()
                print("\nLimpiando memoria...")
                time.sleep(2)

            batch_cols = columns[i:i + batch_size]
            print(f"\nProcesando lote {i//batch_size + 1} de {(total_cols + batch_size - 1)//batch_size}")

            for col in batch_cols:
                try:
                    # Solo variables continuas
                    is_continuous = True

                    # Crear gráfico
                    plot_feature(
                        df=df,
                        col_name=col,
                        isContinuous=is_continuous,
                        target=target_col,
                        max_categories=15,    # Límite de categorías (no aplica aquí)
                        sample_size=50000     # Límite de registros
                    )

                    # Limpiar memoria
                    clear_memory()

                except Exception as e:
                    print(f"Error procesando {col}: {str(e)}")
                    continue

            # Pausa entre lotes
            time.sleep(1)

        print("\nProceso completado exitosamente.")

    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        clear_memory()

    finally:
        clear_memory()



























#NORMALIZATION & SCALING

def normalize_minmax(df):
    # Normalizes numerical columns to the range [0, 1] using Min-Max scaling
    df_normalized = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    print("Min-Max normalization completed.")
    return df_normalized

# 7. ADVANCED FUNCTIONS

def cramers_v(confusion_matrix):
    # Calculates Cramér's V for categorical-categorical association
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('index',axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

def get_percent_null_values_target(pd_loan, list_var_continuous, target):
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum() > 0:
            # Obtener las proporciones de las categorías del target donde la variable es nula
            value_counts = pd_loan[target][pd_loan[i].isnull()].value_counts(normalize=True)
            if not value_counts.empty:
                pd_concat_percent = pd.DataFrame(value_counts).reset_index().T
                
                # Renombrar columnas para evitar errores con índices vacíos
                pd_concat_percent.columns = [f"Category_{k}" for k in range(pd_concat_percent.shape[1])]
                pd_concat_percent = pd_concat_percent.drop('index', axis=0)
                
                # agregar columnas adicionales con información
                pd_concat_percent['variable'] = i
                pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
                pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum() / pd_loan.shape[0]
                pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            else:
                pd_concat_percent = pd.DataFrame({
                    'variable': [i],
                    'sum_null_values': [pd_loan[i].isnull().sum()],
                    'porcentaje_sum_null_values': [pd_loan[i].isnull().sum() / pd_loan.shape[0]]
                })
                pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
    
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final



def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

