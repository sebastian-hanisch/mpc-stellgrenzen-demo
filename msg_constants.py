"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Modell (gilt für jede Zahl der Demo): x⁺ = A x + B u, Wagen A = [[a, 1], [0, 1]], B = (0; 1), Q = I, Stellaufwand R·u², Stellgrenze |u| ≤ u_max,
Kosten ohne Faktor ½ (nur im Skalar-Handbeispiel mit ½, wie bei Rawlings). Keine Zustandsgrenzen, keine Terminalmenge."""

# ---------------------------------------------------------------------------------------------------------------- Regler (Seitenleiste)
N_MIN, N_MAX, DEFAULT_N = 1, 20, 10                       # Horizont; N ≤ 20: schlecht konditioniertes H braucht darüber einen robusten Löser (README)
UMAX_CHOICES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)           # Stellgrenze; nur diese Werte, weil die Referenzkosten (N = 60) für sie vorgerechnet sind
DEFAULT_UMAX = 1.0
R_CHOICES = (0.1, 1.0, 10.0)                              # Stellaufwand; ebenso
DEFAULT_R = 1.0
TERMINALS = ("DARE", "Q")                                 # Endkosten: Riccati-Lösung P (= unbeschränkter unendlicher Horizont) oder P = Q = I
DEFAULT_TERMINAL = "DARE"
TERMINAL_LABELS = {"DARE": "Riccati-Lösung P (DARE)", "Q": "nur Q (P = I)"}
X1_MIN, X1_MAX, DEFAULT_X1, X1_STEP = -12.0, 12.0, 10.0, 0.5
X2_MIN, X2_MAX, DEFAULT_X2, X2_STEP = -4.0, 4.0, 0.0, 0.5

# ---------------------------------------------------------------------------------------------------------------- Regler (Hauptbereich)
E1_MIN, E1_MAX, DEFAULT_E1, E1_STEP = -12.0, 12.0, 10.0, 0.5      # Experiment 1: Startwert des Skalars
SYSTEMS = ("Wagen", "Skalar")
DEFAULT_SYSTEM = "Wagen"
A_CHOICES = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25)             # Experiment 6: offen instabile Strecke; nur diese Werte (vorgerechnet)
DEFAULT_A = 1.2
E6_N_CHOICES = (3, 5, 10, 20)
DEFAULT_E6_N = 20

# ---------------------------------------------------------------------------------------------------------------- Rechnung
SCALAR_UMAX = 1.0                                         # Experiment 1 ist Rawlings Bsp. 2.5: N = 2, |u| ≤ 1, Endkosten ½x², Faktor ½
STEPS_MAX = 400                                           # Regelkreis höchstens so viele Schritte (reicht für |x| < 1e-9 an allen Gitterpunkten)
CONV_TOL = 1e-9                                           # Regelkreis wird bis hierher gefahren
CONV_COUNT_TOL = 1e-6                                     # „konvergiert“ heißt: Endnorm unter dieser Schwelle
EQUAL_REL = 1e-9                                          # Kosten gelten als gleich, wenn sie relativ um höchstens so viel abweichen (zwischen 1e-12 und 1e-6 liegt kein Punkt)
REF_N = 60                                                # Referenz: MPC mit diesem Horizont und Endkosten DARE (kein bewiesener unendlicher Horizont)
GRID_X1 = (-10.0, 10.0, 1.0)                              # Standardgitter der Wagen-Vergleiche: x1 von … bis … Schritt …
GRID_X2 = (-4.0, 4.0, 0.5)                                # 21 × 17 − 1 = 356 Startzustände
FIRSTU_X1 = (-10.0, 10.0, 0.5)                            # Gitter der Erststellsignal-Prüfung (Plan D5): 41 × 33 = 1 353 Punkte
FIRSTU_X2 = (-4.0, 4.0, 0.25)
SCALAR_GRID = (-12.0, 12.0, 0.01)                         # Skalar: 2 401 Punkte
UNSTABLE_GRID = (-6.0, 6.0, 0.5)                          # Experiment 6: 25 × 25 − 1 = 624 Startzustände
UNSTABLE_UMAX = 1.0
UNSTABLE_R = 1.0
CTRL_STEPS = 100                                          # Steuerbarkeit: Stellfolge der Länge 100 mit |u| ≤ 1 bis zum Ursprung (lineares Programm)
SCALING_FACTORS = (0.25, 0.5, 2.0, 4.0, 8.0)              # Experiment 3

# ---------------------------------------------------------------------------------------------------------------- Farben
COLOR_MPC = "#2F6B65"
COLOR_CLIP = "#c0504d"
COLOR_NEUTRAL = "#4c78a8"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_OK = "#2F6B65"
COLOR_BAD = "#c0504d"

# ---------------------------------------------------------------------------------------------------------------- Presets
PRESETS = {
    "Wagen, Grenze 1 (Voreinstellung)": {},
    "Von Hand: Skalar, x = 10": {"e1_x": 10.0, "e2_system": "Skalar", "n": 2, "terminal": "Q"},
    "Abschneiden = MPC im Skalar": {"e2_system": "Skalar", "n": 8},
    "Enge Grenze 0,5": {"umax": 0.5},
    "Zu kurzer Horizont ohne Endkosten": {"n": 2, "terminal": "Q"},
    "Instabile Strecke a = 1,2": {"n": 20, "e6_a": 1.2, "e6_n": 20},
}
PRESET_HELP = {
    "Wagen, Grenze 1 (Voreinstellung)": "Doppelintegrator, Stellgrenze 1, Horizont 10, Endkosten aus der Riccati-Gleichung, Start (10; 0).",
    "Von Hand: Skalar, x = 10": "Das Handbeispiel (Rawlings, Kap. 2): Skalar, Horizont 2, Grenze 1, Endkosten nur Q.",
    "Abschneiden = MPC im Skalar": "Skalar mit Horizont 8: das erste Stellsignal des QP gleicht dem abgeschnittenen unbeschränkten Gesetz.",
    "Enge Grenze 0,5": "Wagen mit Stellgrenze 0,5: das Verhältnis Startabstand zu Grenze wächst, der Abstand zwischen Abschneiden und MPC auch.",
    "Zu kurzer Horizont ohne Endkosten": "Wagen, Horizont 2 und Endkosten nur Q: MPC ist dann in den meisten Startzuständen teurer als Abschneiden.",
    "Instabile Strecke a = 1,2": "Offen instabiler Wagen (a = 1,2), Horizont 20: die Karten zeigen, welche Startzustände konvergieren.",
}
