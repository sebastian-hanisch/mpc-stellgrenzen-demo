"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset ist eindeutig; Verdikt-Zeile je Preset (DEMO-PLAYBOOK §11a Punkt 7)."""

import dataclasses

import msg_constants as C
import msg_evaluation as ev
import msg_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_ranges():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert P.snap(spec, value) == value, (name, key, value)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert ev.Settings(**C.PRESETS[first]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_every_settings_field():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(ev.Settings)}


def test_every_preset_is_distinct():
    """Zwei Presets mit denselben Werten würden sich die Hervorhebung teilen (der erste gewinnt)."""
    voll = [tuple(sorted({**{k: getattr(ev.Settings(), k) for k in P.PRESET_KEYS}, **p}.items())) for p in C.PRESETS.values()]
    assert len(set(voll)) == len(voll)


def test_defaults_are_inside_their_choices():
    for spec in P.SETTING_SPECS.values():
        if spec.choices is not None:
            assert spec.default in spec.choices
        if spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi
