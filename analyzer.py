from functools import lru_cache
from pathlib import Path
import argparse
import json
import math
from collections import defaultdict
import csv
import re


def segment_and_tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[а-яё]+(?:[-'][а-яё]+)*", text)
    return tokens


def lemmatize_tokens(tokens, morph=None):
    if not tokens:
        return []
    morph = morph or get_morph()
    lemmas_with_pos = []
    for token in tokens:
        parsed = morph.parse(token)[0]
        lemma = parsed.normal_form
        part_of_speech = get_part_of_speech(parsed.tag)
        lemmas_with_pos.append((lemma, part_of_speech))
    return lemmas_with_pos


def get_part_of_speech(tag):
    pos_mapping = {
        'NOUN': 'СУЩ',
        'ADJF': 'ПРИЛ',
        'ADJS': 'ПРИЛ',
        'COMP': 'ПРИЛ',
        'VERB': 'ГЛ',
        'INFN': 'ГЛ',
        'PRTF': 'ПРИЧ',
        'PRTS': 'ПРИЧ',
        'GRND': 'ДЕЕП',
        'NUMR': 'ЧИСЛ',
        'ADVB': 'НАР',
        'NPRO': 'МЕСТ',
        'PRED': 'ПРЕДК',
        'PREP': 'ПРЕДЛ',
        'CONJ': 'СОЮЗ',
        'PRCL': 'ЧАСТ',
        'INTJ': 'МЕЖД',
    }
    main_pos = getattr(tag, 'POS', None) or str(tag).split(',')[0].split()[0]
    return pos_mapping.get(main_pos, 'НЕИЗВ')


def remove_noise_words(lemmas_with_pos):
    noise_pos = {'ЧИСЛ', 'МЕСТ', 'ПРЕДЛ', 'СОЮЗ', 'ЧАСТ', 'НЕИЗВ'}
    exceptions = {'не', 'ни'}
    filtered_lemmas = []
    noise_words = []
    for lemma, pos in lemmas_with_pos:
        if pos in noise_pos and lemma not in exceptions:
            noise_words.append((lemma, pos))
            continue
        filtered_lemmas.append((lemma, pos))
    return filtered_lemmas, noise_words


def combine_words(lemmas_with_pos):
    mods = {'не', 'очень', 'крайне', 'необычайно', 'слегка', 'почти', 'чуть', 'чуть-чуть'}
    result = []
    i = 0
    while i < len(lemmas_with_pos):
        lemma = lemmas_with_pos[i][0]
        if lemma in mods:
            phrase = [lemma]
            i += 1
            while i < len(lemmas_with_pos) and lemmas_with_pos[i][0] in mods:
                phrase.append(lemmas_with_pos[i][0])
                i += 1
            if i < len(lemmas_with_pos):
                phrase.append(lemmas_with_pos[i][0])
                result.append(' '.join(phrase))
                i += 1
            else:
                result.extend(phrase)
        else:
            result.append(lemma)
            i += 1
    return result


def load_tone_dict(dict_path='kartaslovsent.csv'):
    tone_dict = {}
    with open(dict_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            term = row['term'].lower()
            value = float(row['value'])
            if not term or not math.isfinite(value):
                raise ValueError('Dictionary terms and values must be nonempty and finite')
            tone_dict[term] = value
    return tone_dict


def calculate_phrase_tones(phrases, tone_dict):
    results = []
    for phrase in phrases:
        words = phrase.split()
        if len(words) == 1:
            tone = tone_dict.get(words[0], 0)
            results.append((phrase, tone))
            continue
        main_word = words[-1]
        main_tone = tone_dict.get(main_word, 0)
        modified_tone = main_tone
        for modifier in reversed(words[:-1]):
            if modifier in {'не', 'ни'}:
                modified_tone *= -1
            elif modifier in {'очень', 'крайне', 'необычайно'}:
                modified_tone *= 2
            elif modifier in {'слегка', 'чуть', 'чуть-чуть', 'почти'}:
                modified_tone *= 0.5
        results.append((phrase, modified_tone))
    return results


def get_tone_category(value):
    if value > 0.33:
        return "положительное"
    elif value < -0.33:
        return "отрицательное"
    else:
        return "нейтральное"


def tone_category_for_text(value):
    if value >= 0.5:
        return "крайне положительный"
    elif value >= 0.1:
        return "положительный"
    elif value > -0.1:
        return "нейтральный"
    elif value > -0.5:
        return "отрицательный"
    else:
        return "крайне отрицательный"


def create_statistics_table(phrases_with_tones, noise_words, tone_dict):

    phrase_stats = defaultdict(lambda: {'count': 0, 'total_tone': 0, 'base_tone': None})

    for phrase, tone in phrases_with_tones:
        words = phrase.split()
        if len(words) == 1:
            base_word = words[0]
            base_tone = tone_dict.get(base_word, 0)
        else:
            base_word = words[-1]
            base_tone = tone_dict.get(base_word, 0)

        phrase_stats[phrase]['count'] += 1
        phrase_stats[phrase]['total_tone'] += tone
        phrase_stats[phrase]['base_tone'] = base_tone

    table_rows = []

    for phrase, stats in phrase_stats.items():
        category = get_tone_category(stats['total_tone'] / stats['count'] if stats['count'] > 0 else 0)
        table_rows.append({
            'Фраза/Слово': phrase,
            'Статус': 'знач.',
            'Количество': stats['count'],
            'Базовая тональность': stats['base_tone'],
            'Суммарная тональность': stats['total_tone'],
            'Категория': category
        })

    noise_word_counts = defaultdict(int)
    for word, pos in noise_words:
        noise_word_counts[(word, pos)] += 1

    for (word, pos), count in noise_word_counts.items():
        table_rows.append({
            'Фраза/Слово': f"{word} ({pos})",
            'Статус': 'шум.',
            'Количество': count,
            'Базовая тональность': 0,
            'Суммарная тональность': 0,
            'Категория': "нейтральное"
        })

    table_rows.sort(key=lambda x: x['Базовая тональность'], reverse=True)
    return table_rows


def save_statistics_table(table_rows, filename):
    with open(filename, 'x', encoding='utf-8') as f:
        f.write(f"{'Фраза/Слово':<30} {'Статус':<6} {'Количество':<10} {'Базовая тональность':<18} "
                f"{'Суммарная тональность':<20} {'Категория':<15}\n")
        f.write("=" * 100 + "\n")
        for row in table_rows:
            f.write(f"{row['Фраза/Слово']:<30} {row['Статус']:<6} {row['Количество']:<10} "
                    f"{row['Базовая тональность']:<18.3f} {row['Суммарная тональность']:<20.3f} {row['Категория']:<15}\n")


def calculate_comprehensive_statistics(tokens, lemmas_with_pos, filtered_lemmas, phrases, tone_results, tone_dict, noise_words):
    stats = {}

    stats['словоупотреблений'] = len(tokens)
    stats['словоформ'] = len(set(tokens))
    stats['лемм'] = len(set(lemma for lemma, pos in lemmas_with_pos))
    stats['значимых_слов'] = len(filtered_lemmas)
    stats['значимых_лемм'] = len(set(lemma for lemma, pos in filtered_lemmas))
    stats['значимых_фраз'] = len(phrases)
    stats['шумовых_слов'] = len(noise_words)

    tones = [tone for _, tone in tone_results]

    stats['слов_value_больше_0'] = sum(1 for tone in tones if tone > 0)
    stats['слов_value_меньше_0'] = sum(1 for tone in tones if tone < 0)
    stats['слов_value_равно_0'] = sum(1 for tone in tones if tone == 0)

    stats['положительных_слов'] = sum(1 for tone in tones if tone > 0.33)
    stats['отрицательных_слов'] = sum(1 for tone in tones if tone < -0.33)
    stats['нейтральных_слов'] = sum(1 for tone in tones if -0.33 <= tone <= 0.33)

    stats['положительность'] = sum(tone for tone in tones if tone > 0)
    stats['отрицательность'] = sum(tone for tone in tones if tone < 0)
    stats['тональность'] = stats['положительность'] + stats['отрицательность']
    stats['тональный_разброс'] = stats['положительность'] - stats['отрицательность']

    return stats


def save_comprehensive_statistics(stats, filename):
    with open(filename, 'x', encoding='utf-8') as f:
        f.write("СТАТИСТИКА АНАЛИЗА ТЕКСТА\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"ОСНОВНЫЕ ПОКАЗАТЕЛИ:\n")
        f.write(f"Словоупотреблений (токенов): {stats['словоупотреблений']}\n")
        f.write(f"Словоформ (различных токенов): {stats['словоформ']}\n")
        f.write(f"Лемм (уникальных после лемматизации): {stats['лемм']}\n")
        f.write(f"Значимых слов (всех после удаления шумовых): {stats['значимых_слов']}\n")
        f.write(f"Значимых лемм (уникальных после удаления шумовых): {stats['значимых_лемм']}\n")
        f.write(f"Значимых фраз (после создания фраз): {stats['значимых_фраз']}\n\n")

        f.write(f"ДОПОЛНИТЕЛЬНЫЕ ПОКАЗАТЕЛИ:\n")
        f.write(f"Слов с value > 0: {stats['слов_value_больше_0']}\n")
        f.write(f"Слов с value < 0: {stats['слов_value_меньше_0']}\n")
        f.write(f"Слов с value = 0: {stats['слов_value_равно_0']}\n")
        f.write(f"Шумовых слов: {stats['шумовых_слов']}\n")
        f.write(f"Положительных слов (value > 0.33): {stats['положительных_слов']}\n")
        f.write(f"Отрицательных слов (value < -0.33): {stats['отрицательных_слов']}\n")
        f.write(f"Нейтральных слов (-0.33 ≤ value ≤ 0.33): {stats['нейтральных_слов']}\n\n")

        f.write(f"СУММАРНЫЕ ПОКАЗАТЕЛИ ТОНАЛЬНОСТИ:\n")
        f.write(f"Положительность (тональность слов с value > 0): {stats['положительность']:.3f}\n")
        f.write(f"Отрицательность (тональность слов с value < 0): {stats['отрицательность']:.3f}\n")
        f.write(f"Тональность (положительность + отрицательность): {stats['тональность']:.3f}\n")
        f.write(f"Тональный разброс (положительность - отрицательность): {stats['тональный_разброс']:.3f}\n\n")

        f.write("=" * 50 + "\n\n")
        f.write("РЕЗУЛЬТАТ:\n")
        f.write(f"Текст: {tone_category_for_text(stats['тональность'] / max(1, stats['значимых_фраз']))}")


def create_plots(stats, filename):
    import matplotlib.pyplot as plt
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    labels1 = ['value > 0', 'value < 0', 'value = 0', 'Шумовые слова']
    sizes1 = [
        stats['слов_value_больше_0'],
        stats['слов_value_меньше_0'],
        stats['слов_value_равно_0'],
        stats['шумовых_слов']
    ]
    colors1 = ['#99ff99', '#ff9999', '#66b3ff', '#ffcc99']
    if sum(sizes1):
        ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', startangle=90)
    else:
        ax1.text(.5, .5, 'Нет данных', ha='center')
    ax1.set_title('Распределение слов по тональности', fontsize=14, fontweight='bold')

    labels2 = ['Положительные', 'Отрицательные', 'Нейтральные']
    sizes2 = [
        stats['положительных_слов'],
        stats['отрицательных_слов'],
        stats['нейтральных_слов']
    ]
    colors2 = ['#99ff99', '#ff9999', '#66b3ff']
    if sum(sizes2):
        ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%', startangle=90)
    else:
        ax2.text(.5, .5, 'Нет данных', ha='center')
    ax2.set_title('Категории тональности значимых слов', fontsize=14, fontweight='bold')

    denominators1 = [
        stats['словоупотреблений'],
        stats['словоформ'],
        stats['лемм']
    ]
    labels3 = ['Словоупотреб.', 'Словоформы', 'Леммы']
    normalized_tones1 = [stats['тональность'] / denom if denom > 0 else 0 for denom in denominators1]
    ax3.axhspan(-1, -0.5, alpha=0.3, color='red', label='Крайне отриц.')
    ax3.axhspan(-0.5, -0.1, alpha=0.3, color='lightcoral', label='Отрицательная')
    ax3.axhspan(-0.1, 0.1, alpha=0.3, color='gray', label='Нейтральная')
    ax3.axhspan(0.1, 0.5, alpha=0.3, color='lightgreen', label='Положительная')
    ax3.axhspan(0.5, 1, alpha=0.3, color='green', label='Крайне полож.')
    bars1 = ax3.bar(labels3, normalized_tones1, color=['#ffcc99', '#ffcc99', '#ffcc99'], alpha=1, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Нормализованная тональность')
    ax3.set_title('Тональность по основным показателям', fontsize=14, fontweight='bold')
    ax3.set_ylim(-max(1, max(map(abs, normalized_tones1))*1.15), max(1, max(map(abs, normalized_tones1))*1.15))
    ax3.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars1, normalized_tones1):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                 fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    denominators2 = [
        stats['значимых_слов'],
        stats['значимых_лемм'],
        stats['значимых_фраз']
    ]
    labels4 = ['Знач. слова', 'Знач. леммы', 'Знач. фразы']
    normalized_tones2 = [stats['тональность'] / denom if denom > 0 else 0 for denom in denominators2]
    ax4.axhspan(-1, -0.5, alpha=0.3, color='red', label='Крайне отриц.')
    ax4.axhspan(-0.5, -0.1, alpha=0.3, color='lightcoral', label='Отрицательная')
    ax4.axhspan(-0.1, 0.1, alpha=0.3, color='gray', label='Нейтральная')
    ax4.axhspan(0.1, 0.5, alpha=0.3, color='lightgreen', label='Положительная')
    ax4.axhspan(0.5, 1, alpha=0.3, color='green', label='Крайне полож.')
    bars2 = ax4.bar(labels4, normalized_tones2, color=['#ffcc99', '#ffcc99', '#ffcc99'], alpha=1, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Нормализованная тональность')
    ax4.set_title('Тональность по значимым показателям', fontsize=14, fontweight='bold')
    ax4.set_ylim(-max(1, max(map(abs, normalized_tones2))*1.15), max(1, max(map(abs, normalized_tones2))*1.15))
    ax4.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars2, normalized_tones2):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                 fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()




@lru_cache(maxsize=1)
def get_morph():
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError:
        try:
            from pymorphy2 import MorphAnalyzer
        except ImportError as error:
            raise RuntimeError('Install pymorphy3 to use morphological analysis') from error
    return MorphAnalyzer()


def analyze_text(text, tone_dict, morph=None):
    tokens, lemmas, filtered, phrases, noise = [], [], [], [], []
    for sentence in re.split(r'[.!?;,:\n]+', text):
        current_tokens = segment_and_tokenize(sentence)
        current_lemmas = lemmatize_tokens(current_tokens, morph)
        current_filtered, current_noise = remove_noise_words(current_lemmas)
        tokens.extend(current_tokens)
        lemmas.extend(current_lemmas)
        filtered.extend(current_filtered)
        noise.extend(current_noise)
        phrases.extend(combine_words(current_filtered))
    tones = calculate_phrase_tones(phrases, tone_dict)
    stats = calculate_comprehensive_statistics(tokens, lemmas, filtered, phrases, tones, tone_dict, noise)
    return {'tokens': tokens, 'lemmas': lemmas, 'phrases': tones,
            'statistics': stats, 'table': create_statistics_table(tones, noise, tone_dict)}


def main():
    parser = argparse.ArgumentParser(description='Морфологический и словарный анализ русскоязычного текста')
    parser.add_argument('text', type=Path)
    parser.add_argument('--dictionary', type=Path, default=Path(__file__).parent / 'examples' / 'sentiment.csv')
    parser.add_argument('--output-dir', type=Path, help='Новая папка для результатов')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    if args.plots and args.output_dir is None:
        parser.error('--plots requires --output-dir')
    try:
        text = args.text.read_text(encoding='utf-8-sig')
        result = analyze_text(text, load_tone_dict(args.dictionary))
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            save_statistics_table(result['table'], args.output_dir / 'table.txt')
            save_comprehensive_statistics(result['statistics'], args.output_dir / 'statistics.txt')
            with (args.output_dir / 'analysis.json').open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
            if args.plots:
                create_plots(result['statistics'], args.output_dir / 'plots.png')
        print(json.dumps(result['statistics'], ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
