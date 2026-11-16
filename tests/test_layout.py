"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import msg_constants as C
import msg_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    """Der Regler mit diesem Schlüssel: Seitenleiste oder Hauptbereich (Experiment-Regler stehen im Hauptbereich und gehören trotzdem zu SETTING_SPECS)."""
    for bereich in (at.sidebar, at.main):
        for art in REGLER_ARTEN:
            try:
                return getattr(bereich, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} steht weder in der Seitenleiste noch im Hauptbereich")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_steht_in_setting_specs_und_umgekehrt():
    """Jeder Regler der App (Seitenleiste und Hauptbereich) steht in SETTING_SPECS, sonst hebt seine Änderung das aktive Preset nicht auf; jeder Eintrag hat einen Regler."""
    at = _app()
    schluessel = {w.key for bereich in (at.sidebar, at.main) for art in REGLER_ARTEN for w in getattr(bereich, art) if w.key}
    assert schluessel == set(P.SETTING_SPECS), (sorted(schluessel - set(P.SETTING_SPECS)), sorted(set(P.SETTING_SPECS) - schluessel))


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    kandidaten = [c for c in spec.choices if c != regler.value] if spec.choices is not None else [spec.hi, spec.lo, (spec.lo + spec.hi) / 2]
    # ein Wert, mit dem die Einstellungen (Standard plus diese Änderung) zufällig einem anderen Preset entsprechen, ist kein Test der Aufhebung
    neu = next(v for v in kandidaten if v != regler.value and not _entspricht_preset(state_key, v))
    regler.set_value(neu).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def _entspricht_preset(state_key, wert) -> bool:
    """Entsprechen Standardwerte mit dem einen geänderten Regler genau einem Preset?"""
    aktuell = {k: P.SETTING_SPECS[sk].default for k, sk in P.PRESET_KEYS.items()}
    aktuell[next(k for k, sk in P.PRESET_KEYS.items() if sk == state_key)] = wert
    return any(all(aktuell[k] == preset.get(k, P.SETTING_SPECS[P.PRESET_KEYS[k]].default) for k in aktuell) for preset in C.PRESETS.values())


def test_preset_setzt_alle_regler_auf_seine_werte():
    """Ein angeklicktes Preset setzt jeden Regler auf den Wert des Presets oder auf den Standard (so bleibt die Hervorhebung eindeutig)."""
    for name, preset in C.PRESETS.items():
        at = _app()
        at.sidebar.button(key=f"preset_{name}").click().run()
        assert not at.exception, (name, [str(e) for e in at.exception])
        for key, state_key in P.PRESET_KEYS.items():
            erwartet = preset.get(key, P.SETTING_SPECS[state_key].default)
            assert _regler(at, state_key).value == erwartet, (name, key)
