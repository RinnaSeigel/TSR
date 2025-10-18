# %% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from scipy import stats
import scipy.stats as sps
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [1-2] ЗАГРУЗКА ДАННЫХ И СТАТИСТИКА

def load_and_analyze_data():
    """Загрузка данных и базовый анализ"""
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    selected_features = ['sepal length (cm)', 'sepal width (cm)',
                        'petal length (cm)', 'petal width (cm)']
    df_filtered = df[selected_features + ['species']].dropna()
    print("=" * 60)
    print("СТАТИСТИКА ДАТАСЕТА IRIS")
    print("=" * 60)
    print(f"Размерность исходного датасета: {df.shape}")
    print(f"Размерность отфильтрованного датасета: {df_filtered.shape}")
    print(f"Количество признаков: {len(selected_features)}")
    print(f"Количество классов: {len(df_filtered['species'].unique())}")
    print("\nРаспределение по классам:")
    class_counts = df_filtered['species'].value_counts()
    for species, count in class_counts.items():
        print(f" - {species}: {count} объектов")
    null_count = df.isnull().sum().sum()
    null_percentage = (null_count / df.size * 100)
    print(f"\nПропущенные значения: {null_count} ({null_percentage:.2f}%)")
    print(f"Удалено объектов при фильтрации: {len(df) - len(df_filtered)}")
    print("\nОсновные статистики отфильтрованного датасета:")
    stats_table = df_filtered[selected_features].describe()
    print(stats_table.to_string())
    return df_filtered, selected_features

df, features = load_and_analyze_data()
classes = df['species'].unique()

# %% [3] ВИЗУАЛИЗАЦИЯ ДАТАСЕТА

def create_pair_plots(df, features, classes):
    class_markers = {'setosa': 'o', 'versicolor': 's', 'virginica': '^'}
    class_colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle('Матрица парных графиков переменных с разными классами', fontsize=16, y=0.92)
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            ax = axes[i, j]
            if i != j:
                for class_name in classes:
                    class_data = df[df['species'] == class_name]
                    ax.scatter(class_data[feat2], class_data[feat1],
                               marker=class_markers[class_name],
                               color=class_colors[class_name],
                               alpha=0.7, label=class_name, s=50)
                ax.set_xlabel(feat2, fontsize=10)
                ax.set_ylabel(feat1, fontsize=10)
                ax.grid(True, alpha=0.3)
                if i == 0 and j == len(features)-1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                for class_name in classes:
                    class_data = df[df['species'] == class_name]
                    ax.hist(class_data[feat1], bins=25, alpha=0.7,
                            label=class_name, color=class_colors[class_name],
                            edgecolor='black')
                ax.set_xlabel(feat1, fontsize=10)
                ax.set_ylabel('Частота', fontsize=10)
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend()
    plt.tight_layout()
    plt.savefig('pair_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
create_pair_plots(df, features, classes)

# %% [4] КОРРЕЛЯЦИОННЫЙ АНАЛИЗ

def calculate_correlations(data, features, method='pearson'):
    corr_matrix = np.zeros((len(features), len(features)))
    p_matrix = np.zeros((len(features), len(features)))
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if method == 'pearson':
                corr, p_value = stats.pearsonr(data[feat1], data[feat2])
            else:
                corr, p_value = stats.spearmanr(data[feat1], data[feat2])
            corr_matrix[i, j] = corr
            p_matrix[i, j] = p_value
    return pd.DataFrame(corr_matrix, index=features, columns=features), \
           pd.DataFrame(p_matrix, index=features, columns=features)

def plot_correlation_heatmaps(corr_matrix, p_matrix, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                ax=ax1, fmt='.3f', cbar_kws={'label': 'Коэффициент корреляции'})
    ax1.set_title(f'{title} - Коэффициенты корреляции')
    sns.heatmap(p_matrix, annot=True, cmap='viridis',
                ax=ax2, fmt='.4f', cbar_kws={'label': 'p-value'})
    ax2.set_title(f'{title} - P-значения')
    plt.tight_layout()
    plt.show()

print("КОРРЕЛЯЦИИ ДЛЯ ВСЕГО ДАТАСЕТА")
print("=" * 50)
pearson_corr, pearson_p = calculate_correlations(df, features, 'pearson')
spearman_corr, spearman_p = calculate_correlations(df, features, 'spearman')
print("Корреляция Пирсона:")
print(pearson_corr.to_string())
print("\nP-значения Пирсона:")
print(pearson_p.to_string())
plot_correlation_heatmaps(pearson_corr, pearson_p, 'Пирсон (весь датасет)')
plot_correlation_heatmaps(spearman_corr, spearman_p, 'Спирман (весь датасет)')

print("\n" + "="*60)
print("КОРРЕЛЯЦИИ ПО КЛАССАМ")
print("="*60)
for class_name in classes:
    class_data = df[df['species'] == class_name]
    pearson_class, pearson_p_class = calculate_correlations(class_data, features, 'pearson')
    spearman_class, spearman_p_class = calculate_correlations(class_data, features, 'spearman')
    print(f"\n{class_name.upper()}:")
    print("Корреляция Пирсона:")
    print(pearson_class.to_string())
    plot_correlation_heatmaps(pearson_class, pearson_p_class, f'Пирсон ({class_name})')
    plot_correlation_heatmaps(spearman_class, spearman_p_class, f'Спирман ({class_name})')

# %% [5-6] ЛИНЕЙНАЯ РЕГРЕССИЯ С ИНТЕРВАЛАМИ
# ... (оставьте ваш код, если оно нужно!)

# %% [8-9] ДИСПЕРСИОННЫЙ АНАЛИЗ (ANOVA) + RAINCLOUD PLOT

def create_pretty_raincloud_plot(df, feature, classes):
    class_palette = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
    class_order = list(classes)
    plt.figure(figsize=(10, 6))
    ax = pt.RainCloud(x='species', y=feature, data=df,
                      palette=class_palette,
                      order=class_order,
                      width_viol=0.6,
                      width_box=0.3,
                      alpha=0.4,
                      jitter=True,
                      move=0.2,
                      point_size=2,
                      box_showfliers=True,
                      bw=0.3)
    plt.title(f"Raincloud plot для '{feature}'", fontsize=16)
    plt.xlabel("Класс", fontsize=13)
    plt.ylabel(feature, fontsize=13)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"raincloud_{feature.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=300)
    plt.show()

anova_results = []
for feature in features:
    print(f"\nДИСПЕРСИОННЫЙ АНАЛИЗ ДЛЯ ПРИЗНАКА: {feature}")
    data_groups = [df[df['species'] == cls][feature] for cls in classes]
    f_stat, p_value = stats.f_oneway(*data_groups)
    print(f"F-критерий = {f_stat:.4f}")
    print(f"P-value = {p_value:.4f}")
    tukey = pairwise_tukeyhsd(df[feature], df['species'], alpha=0.05)
    print(tukey.summary())
    create_pretty_raincloud_plot(df, feature, classes)
    anova_results.append({
        'Признак': feature,
        'F-статистика': f_stat,
        'P-value': p_value,
        'Степени свободы (межгрупповые)': len(classes) - 1,
        'Степени свободы (внутригрупповые)': len(df) - len(classes)
    })

anova_summary = pd.DataFrame(anova_results)
print("\nСВОДНАЯ ТАБЛИЦА ANOVA ДЛЯ ВСЕХ ПРИЗНАКОВ")
print("=" * 60)
print(anova_summary.to_string(index=False))

# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
with open('lab_results_summary.txt', 'w', encoding='utf-8') as f:
    f.write("РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ РАБОТЫ\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Датасет: Iris\n")
    f.write(f"Размерность: {df.shape}\n")
    f.write(f"Признаки: {', '.join(features)}\n")
    f.write(f"Классы: {', '.join(classes)}\n\n")
    f.write("АНАЛИЗ ANOVA:\n")
    for result in anova_results:
        f.write(f"{result['Признак']}: F = {result['F-статистика']:.4f}, p = {result['P-value']:.4f}\n")

print("Все результаты сохранены в файлы:")
print("- pair_plots.png")
print("- raincloud_*.png (для каждого признака)")
print("- lab_results_summary.txt")

# ФИНАЛЬНАЯ СВОДКА
print("\n" + "="*60)
print("ФИНАЛЬНАЯ СВОДКА РЕЗУЛЬТАТОВ")
print("="*60)
print(f"✓ Проанализировано признаков: {len(features)}")
print(f"✓ Выполнено ANOVA тестов: {len(anova_results)}")
print(f"✓ Создано визуализаций: {4 + len(features)}")
significant_features = [res['Признак'] for res in anova_results if res['P-value'] < 0.05]
print(f"✓ Статистически значимые различия между классами найдены для {len(significant_features)} признаков:")
for feat in significant_features:
    print(f" - {feat}")
