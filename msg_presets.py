"""Permalink (Begrenzen und Einrasten), Presets („Schnellstart“) und aktives Preset (Standardmuster aus dem Demo-Portfolio).

Jeder Regler der App (Seitenleiste und Hauptbereich) steht in SETTING_SPECS: ein Preset ist aktiv, wenn alle Regler seinen Werten entsprechen,
und die Änderung eines beliebigen Reglers hebt die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""

from dataclasses import dataclass

import streamlit as st

import msg_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # Auswahlregler: nur diese Werte; ein anderer Wert aus der Adresszeile rastet auf den nächsten ein


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "umax_slider": SettingSpec("umax", float, C.DEFAULT_UMAX, choices=C.UMAX_CHOICES),
    "r_slider": SettingSpec("r", float, C.DEFAULT_R, choices=C.R_CHOICES),
    "terminal": SettingSpec("term", str, C.DEFAULT_TERMINAL, choices=C.TERMINALS),
    "x1_slider": SettingSpec("x1", float, C.DEFAULT_X1, C.X1_MIN, C.X1_MAX),
    "x2_slider": SettingSpec("x2", float, C.DEFAULT_X2, C.X2_MIN, C.X2_MAX),
    "e1_x": SettingSpec("e1", float, C.DEFAULT_E1, C.E1_MIN, C.E1_MAX),
    "e2_system": SettingSpec("sys", str, C.DEFAULT_SYSTEM, choices=C.SYSTEMS),
    "e6_a": SettingSpec("a", float, C.DEFAULT_A, choices=C.A_CHOICES),
    "e6_n": SettingSpec("n6", int, C.DEFAULT_E6_N, choices=C.E6_N_CHOICES),
}
PRESET_KEYS = {"n": "n_slider", "umax": "umax_slider", "r": "r_slider", "terminal": "terminal", "x1": "x1_slider", "x2": "x2_slider",
               "e1_x": "e1_x", "e2_system": "e2_system", "e6_a": "e6_a", "e6_n": "e6_n"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap(spec: SettingSpec, value):
    """Wert auf die Reglergrenzen begrenzen bzw. auf den nächsten erlaubten Wert einrasten."""
    if spec.choices is not None:
        if value in spec.choices:
            return value
        if isinstance(value, (int, float)):
            return min(spec.choices, key=lambda c: abs(c - value))
        return spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt bzw. eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = snap(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset legt alle Regler fest; nicht genannte Regler gehen auf ihren Standardwert (so ist das aktive Preset eindeutig)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos, jeder Regler aus SETTING_SPECS zählt)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )
