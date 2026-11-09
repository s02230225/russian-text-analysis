from types import SimpleNamespace
import pytest

from analyzer import analyze_text, calculate_phrase_tones, combine_words, get_part_of_speech, segment_and_tokenize, save_comprehensive_statistics


class Morph:
    def parse(self, word):
        pos = 'PRCL' if word in {'не', 'ни'} else 'NOUN'
        return [SimpleNamespace(normal_form=word, tag=SimpleNamespace(POS=pos))]


def test_russian_tokens_and_hyphen():
    assert segment_and_tokenize("Ёж, чуть-чуть! 123 ' ") == ['ёж', 'чуть-чуть']


def test_tag_pos_attribute():
    assert get_part_of_speech(SimpleNamespace(POS='VERB')) == 'ГЛ'


@pytest.mark.parametrize('words,expected', [(['хорошо'], 1), (['не', 'хорошо'], -1), (['очень', 'хорошо'], 2), (['слегка', 'хорошо'], .5), (['не', 'не', 'хорошо'], 1)])
def test_modifiers(words, expected):
    phrases = combine_words([(word, 'НАР') for word in words])
    assert calculate_phrase_tones(phrases, {'хорошо': 1}) == [(' '.join(words), expected)]


def test_sentence_boundary_does_not_propagate_negation():
    result = analyze_text('не. хорошо', {'хорошо': 1}, Morph())
    assert ('хорошо', 1) in result['phrases']


def test_empty_text_has_finite_statistics(tmp_path):
    result = analyze_text('', {})
    assert result['statistics']['значимых_фраз'] == 0
    save_comprehensive_statistics(result['statistics'], tmp_path / 'stats.txt')
    assert 'нейтральный' in (tmp_path / 'stats.txt').read_text(encoding='utf-8')
