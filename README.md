# Stellgrenzen: MPC als quadratisches Programm – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-mpc-stellgrenzen-demo.streamlit.app/)**

Diese Demo gehört zum Thema **Regelung (von LQR bis MPC)** auf der Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie baut auf
[`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo) und [`horizont-falle-demo`](https://github.com/sebastian-hanisch/horizont-falle-demo) auf und hebt eine Annahme auf: Das Gesetz u = −K x und der Satz „mit Endkosten aus der Riccati-Gleichung ist der Gewinn für jeden
Horizont derselbe wie der unendliche“ setzen eine **unbegrenzte Stellgröße** voraus. Hier gilt |u| ≤ u_max, und gefragt wird, was MPC als quadratisches Programm (QP) daraus macht. Vehikel ist der **Wagen**
(Doppelintegrator, Ts = 1) und der Skalar x⁺ = x + u als Handbeispiel. Nachbarn zum Thema Regelung: [`plan-stoerung-demo`](https://github.com/sebastian-hanisch/plan-stoerung-demo) (Plan und Störung),
[`lqr-reserven-demo`](https://github.com/sebastian-hanisch/lqr-reserven-demo) (Reserven), [`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo) (die Gegenfrage mit einem PI-Regler und Back-Calculation),
[`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo) (Zustandsgrenzen, Zulässigkeit), [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo) (Sollwert und Störung) und
[`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo) (Rechenzeit als Verzug). Andere Stellen im Portfolio: [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) (dort macht die Klemmung u ≥ 0 eine
lineare Lagerregel chaotisch; hier ist das Lager nicht das Vehikel, und am Doppelintegrator konvergiert das Abschneiden in allen gemessenen Fällen (B12), bei der offen instabilen Strecke dagegen nicht überall (B11)), [`rolling-horizon-demo`](https://github.com/sebastian-hanisch/rolling-horizon-demo) (rollierend planen mit anderem Modell),
[`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) und [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (Rückkopplung im Lager, anderes Modell).
Andere QP-Löser (ADMM, Warmstart, Innere-Punkte-Verfahren) gehören zum Thema Konvexe und konische Optimierung; hier gibt es nur die Aktivmengen-Methode.

## Warum dieses Problem

Am Wagen aus dem Startzustand (10; 0) verlangt das Gesetz u = −K x mit K = (0,42208; 1,24393) (R = 1) die Stellgröße u = −4,22; bei der Grenze 1 geht das nicht. Zwei Wege führen weiter: **„LQR, dann abschneiden“**
(u = clip(−K x)) und **MPC**, das in jedem Takt ein QP mit Boxgrenzen löst. **Wann reicht das Abschneiden, wann nicht, und wie lang muss der Horizont sein?**
Gemessen in diesem Modell (Wagen mit Q = I und R ∈ {0,1; 1; 10}, Endkosten aus der Riccati-Gleichung, nur Stellgrenzen): Im Skalar und bei Horizont N = 1 gibt es keinen Unterschied; am Wagen kann MPC ab N = 2 billiger sein, sofern die Grenze auf dem Gitter greift; bei u_max = 1 und N = 10 ist es in 146 von
356 Startzuständen billiger (größter Vorteil 3,09 %), bei u_max = 0,5 in 294 von 356 (59,41 %), nirgends teurer. Bei der engeren Grenze, also dem größeren Verhältnis Startabstand zu Grenze, ist der Unterschied größer (B5, B7). Bei einer offen instabilen Strecke
(a = 1,2) konvergiert „LQR, dann abschneiden“ in 332 von 624 Startzuständen, MPC mit N = 20 in 494; das sind genau die 494, die sich per linearem Programm in 100 Schritten mit |u| ≤ 1 auf den Ursprung steuern lassen.
Alles ist **Lehrbuchstoff, gemessen in einem wachsenden Beispiel**; eine Neuheitsaussage gibt es nicht, und die Aussagen gelten für die gemessenen Gitter und Parameter, nicht für jede Strecke.

## Modell

**Zeitlage und Kosten (gilt für jede Zahl):** x⁺ = A x + B u mit einem Eingang. Wagen: A = [[a, 1], [0, 1]], B = (0; 1), a = 1 ist der Doppelintegrator, a > 1 offen instabil. Skalar: A = B = 1. Q = I, Stellaufwand R·u²,
Stellgrenze −u_max ≤ u ≤ u_max, **keine Zustandsgrenzen und keine Terminalmenge** (die gehören zu [`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo)). Kosten eines Laufs: Σ (x'Qx + R·u²), gefahren bis |x| < 10⁻⁹
(höchstens 400 Takte; mit „konvergiert“ ist die Endnorm unter 10⁻⁶ gemeint), **ohne Faktor ½** (Konvention von Borrelli/Bemporad/Morari und Åström/Murray); nur das Skalar-Handbeispiel rechnet mit ½ wie Rawlings, damit 123,5 vergleichbar bleibt.
Horizont N, Endkosten x_N'P x_N mit P aus der Riccati-Gleichung (DARE) oder, in Experiment 4 und im Schnellstart „Zu kurzer Horizont ohne Endkosten“, P = Q.

- **Das QP:** Setzt man die Zustände ein, bleibt J = c + ½ u'H u + f'u in der Stellfolge u = (u₀ … u_{N−1}) mit H = 2(Γ'Q̄Γ + R·I), f = F x₀, F = 2 Γ'Q̄Φ, Q̄ = blockdiag(Q, …, Q, P). H ist fest, nur f wechselt mit dem Zustand.
  Gleichungen in eigener Herleitung (Formeln im Expander 📐 der App); geprüft gegen die direkt summierte Kostenfunktion (Differenzen in Brüchen, `tests/test_qp.py`).
- **Zwei Arten „Abschneiden“, getrennt benannt:** (1) **LQR-Gesetz abschneiden**: u = clip(−K x) mit dem Riccati-Gewinn K des unendlichen Horizonts; sie liefert die Kosten der Vergleiche. (2) **Unbeschränkte N-Schritt-Folge abschneiden**: das erste Element der
  unbeschränkten Lösung des N-Schritt-Problems, abgeschnitten; sie dient nur dem Vergleich des ersten Stellsignals mit dem QP (mit Endkosten DARE gleich der ersten Art).
- **Die Referenz** ist MPC mit N = 60 und Endkosten DARE. Sie ist **kein bewiesener unendlicher Horizont**; gestützt wird sie durch den Vergleich von vorhergesagtem und gefahrenem Wert (B9) und im Skalar durch dynamische Programmierung (B2).
- **Keine Zufallsgröße in der App** (Gitter und feste Startzustände), also kein Zufalls-Seed-Regler. Die Gegenproben mit Zufallsinstanzen (nur `tools/make_frozen.py` und die Tests) benutzen den ganzzahligen SplitMix64 aus `msg_rng.py`, Zufalls-Seed 14.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Aktivmengen-Methode** (`msg_qp.qp_active_set`) | primal, nur Boxgrenzen, Kaltstart bei u = 0: Newton-Schritt auf den freien Stellen, Grenze ins Arbeitsset bei Berührung, bei Schritt null entscheidet das Vorzeichen des Gradienten, welche Grenze freigegeben wird; Toleranzen relativ zum Boxradius und zur Gradientengröße | bricht bei schlecht konditioniertem H ohne Abbruch ab (B15); dann Rückfall auf `bvls` (`msg_qp.qp_solve`) |
| **Regelkreis** (`msg_dynamics.run_loop`, `run_clip_batch`) | Schritt für Schritt, in jedem Takt ein QP; „Abschneiden“ vektorisiert über die Startzustände | höchstens 400 Takte, Kosten bis \|x\| < 10⁻⁹ |
| **Exakte Handrechnung** (`msg_exact.py`) | Brüche, alle 3^N Muster aufgezählt (N = 2) | nur für die Handbeispiele |
| **Vorgerechnet** (`data/msg_frozen_*.json`, erzeugt von `tools/make_frozen.py`) | Referenzkosten (N = 60), Horizonttabellen, Aufwand, Gegenprobe, große Zustände, Karten der instabilen Strecke, Orakel-Tabelle, Rückfall | in der App als „vorgerechnet“ gekennzeichnet |

Die Live-Rechnung der App braucht für die 356 Startzustände (zwei Regeln) beim ersten Aufruf je nach Horizont etwa 1 bis 5 Sekunden (Messung auf dem Entwicklungsrechner, nicht festgeschrieben, ohne Zeichnen), danach liegen die Ergebnisse im Zwischenspeicher;
ein QP, ein Regelkreis und die Erststellsignal-Prüfung brauchen deutlich unter einer Sekunde. Die Regler sind begrenzt (N ≤ 20, u_max und R nur aus den Werten, für die die Referenz vorgerechnet ist).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Wagen mit Q = I, nur Stellgrenzen, keine Störung, Gitter wie genannt; „vorgerechnet“ heißt gespeicherte Messung (`data/`), „neu gerechnet“ heißt im Test erneut gerechnet und als Band verglichen (Zählungen ± 2, Gleitkomma relativ 10⁻⁶ bis 10⁻⁹,
Iterationszahlen ± 20 %). Die Spalte „Test“ nennt die Funktion in `tests/test_claims.py` (sonst die Datei). Standardgitter: 356 Startzustände, x₁ von −10 bis 10 (Schritt 1), x₂ von −4 bis 4 (Schritt 0,5).

| Nr | Frage | Befund | Test |
|---|---|---|---|
| B1 | Skalar von Hand (Rawlings Bsp. 2.5: N = 2, Grenze 1, Endkosten ½x², Faktor ½) | u₀ = −sat(3x/5) an 161 von 161 Bruch-Startwerten (x = k/10, k = −80 bis 80); V_N⁰(10) = 247/2 = 123,5 (die Quelle nennt nach Quellenbericht 124, im Original nicht gelesen; 124 ist 123,5 gerundet); der Regelkreis x⁺ = x − sat(3x/5) kostet bis ins Unendliche 8287/42 = 197,3095 (Brüche): die Voraussage liegt 37 % unter den gefahrenen Kosten | `test_b1_*`, `tests/test_exact.py` <!-- lint-ok: dezimalpunkt --> |
| B2 | Skalar: Abschneiden gegen MPC | Erstes Stellsignal des QP gegen das erste Element der abgeschnittenen unbeschränkten Folge: 0 Abweichungen über 10⁻⁹ auf 2 401 Punkten (x von −12 bis 12, Schritt 0,01), N = 1 bis 8, beide Endkosten (größte Abweichung unter 10⁻¹², vorgerechnet), ebenso auf einem zweiten Gitter (Schritt 0,013, Grenze 0,7; 1 847 Punkte). Gefahrene Kosten ab x = 10 (Faktor ½): MPC N = 2 197,3095, LQR-abgeschnitten 197,3090, MPC N = 40 197,3090, dynamische Programmierung (Gitter 0,005) 197,3090; MPC N = 2 liegt 0,0003 % über dem Optimum, weil Endkosten ½x² den Gewinn 3/5 statt 1/φ = 0,618 liefern. **Warum im Skalar kein Unterschied entsteht, ist nicht erklärt**, nur gemessen | `test_b2_*` (vorgerechnet und neu gerechnet) |
| B3 | Wagen, N = 1 | MPC = Abschneiden: gleiche Kosten an 356 von 356 Startzuständen (u_max = 1, R = 1); in Brüchen an 117 von 117 ganzzahligen Zuständen für zwei verschiedene P | `test_b3_*`, `tests/test_exact.py` |
| B4 | Wagen, ab N = 2: erstes Stellsignal | Mit Endkosten DARE ist der unbeschränkte erste Gewinn für N = 1 bis 10 gleich K (größte Abweichung 5,6·10⁻¹⁵ bei R = 1, 2,1·10⁻¹⁴ bei R = 0,1); ohne Endkosten nicht (0,74 bei N = 1, 0,24 bei N = 2). Im Fenster x₁ ∈ [−3, 3], x₂ ∈ [−1,5; 1,5] (N = 10, u_max = 1; 325 Gitterpunkte, nicht ein Gebiet ohne aktive Grenze) liefert das QP −K x an 157 von 325 Gitterpunkten; sobald die Grenze in der Folge greift, weicht das erste Stellsignal von clip(−K x) an 88 von 1 353 Gitterpunkten ab (6,5 %), um höchstens 0,307 | `test_b4_*` |
| B5 | Wagen, Kosten: MPC N = 10 gegen LQR-abgeschnitten (gefahrene Kosten bis \|x\| < 10⁻⁹) | u_max = 1, R = 1: MPC billiger in 146 von 356, gleich in 210, teurer in 0, größter Vorteil 3,09 %; u_max = 0,5, R = 1: 294 / 62 / 0, 59,41 %; u_max = 1, R = 0,1: 234 / 122 / 0, 16,05 %. Bei N = 2 (u_max = 1, R = 1): 142 / 214 / 0; die Aussage „ab N = 2 billiger“ gilt nur, wenn die Grenze auf dem Gitter greift: bei u_max = 2 (R = 1) und bei R = 10 mit u_max = 1 ist MPC in 0 von 356 billiger (N = 2 und N = 10), bei R = 10 und u_max = 0,75 in 30 von 356. Aufschlag des Abschneidens gegen die Referenz (Median / 90-%-Quantil / Maximum): 0,00 / 0,35 / 3,19 %; 1,43 / 10,17 / 146,37 %; 0,08 / 3,34 / 19,11 %. Die Zählung hängt nicht an der Schwelle: zwischen relativ 10⁻¹² und 10⁻⁶ liegt kein Punkt, der kleinste Unterschied darüber ist 2,5·10⁻⁶ (u_max = 1) bzw. 1,8·10⁻⁵ (u_max = 0,5) | `test_b5_*` |
| B6 | Gegenprobe der Absolutaussage „nie teurer“ | 1 000 Zufallsinstanzen (R ∈ {0,1; 1; 10}, u_max ∈ [0,2; 2], x₀ ∈ [−30, 30] × [−12, 12], Zufalls-Seed 14, 3 000 Takte, Endkosten DARE): MPC mit N = 2, 3 und 5 in 0 von 1 000 teurer als Abschneiden, gleich in 232 / 193 / 193, billiger in 768 / 807 / 807, größter Vorteil 72,7 / 74,9 / 88,2 %. Eine Garantie ist das **nicht**: mit Endkosten nur Q ist MPC teurer (B10) | `test_b6_*` |
| B7 | Nur das Verhältnis zählt | Kosten von (c·x₀, c·u_max) sind c² mal die von (x₀, u_max) (Abschneiden und MPC N = 10, 4 Startpunkte × 4 Faktoren; größte relative Abweichung 2,2·10⁻¹⁶). Folge: u_max = 0,5 auf [−10, 10] × [−4, 4] ist u_max = 1 auf [−20, 20] × [−8, 8] | `test_b7_*` |
| B8 | Horizontbedarf (größter Aufschlag von MPC gegen die Referenz, Endkosten DARE) | u_max = 1, R = 1: N = 1: 3,19 %, N = 2: 0,70 %, N = 3: 0,05 %, ab N = 4 unter 0,005 %. u_max = 0,5, R = 1: 146,37 / 34,77 / 7,78 / 1,50 / 0,59 % für N = 1 bis 5, 0,02 % bei N = 7, unter 0,005 % bei N = 10 und 20. u_max = 1, R = 0,1: 19,11 / 2,37 / 0,24 % für N = 1 bis 3, 0,0053 % bei N = 4, ab N = 5 unter 0,005 %. Unter 0,005 % liegt der Aufschlag bei u_max = 1 (R = 1) ab N = 4, bei u_max = 0,5 (doppeltes Verhältnis Zustand zu Grenze) erst bei den gemessenen N = 10 und 20 (N = 8 und 9 nicht gemessen), bei R = 0,1 ab N = 5 | `test_b8_*` |
| B9 | Voraussage gegen gefahren (u_max = 1, R = 1, Endkosten DARE) | x₀ = (10; 0): V₂ = 374,8781, gefahren 375,4336 (+0,15 %); V₅ = V₁₀ = V₆₀ = gefahren = 375,4336. x₀ = (−10; 4): V₂ = 179,5462, gefahren 179,8218, ab N = 5 alle gleich 179,8218. Für (5; −2) und (2; 1) sind V und gefahren bei den gemessenen N = 2, 5, 10, 60 gleich | `test_b9_*` |
| B10 | Ohne Endkosten (P = Q, u_max = 1, R = 1) | N = 1 konvergiert in 12 von 356 Startzuständen; N = 2 in 356, Aufschlag gegen die Referenz Median 0,73 %, Maximum 10,15 %, **MPC teurer als Abschneiden in 322 von 356**, billiger in 34; N = 3: Maximum 0,26 %, teurer in 238; N = 5: Maximum 0,02 %, teurer in 210 | `test_b10_*` |
| B11 | Offen instabile Strecke (A = [[a, 1], [0, 1]], u_max = 1, R = 1, Endkosten DARE, Gitter [−6, 6]², 624 Startzustände, Konvergenz heißt Norm unter 10⁻⁶ nach höchstens 400 Takten ohne 10⁶ zu überschreiten) | **a = 1,2:** Abschneiden 332, MPC N = 3 / 5 / 10 / 20: 378 / 414 / 462 / 494. Per linearem Programm (Stellfolge der Länge 100 mit \|u\| ≤ 1 bis x₁₀₀ = 0) sind 494 Punkte steuerbar; MPC N = 20 konvergiert in genau diesen 494 (0 Abweichungen), Abschneiden verfehlt 162 steuerbare Punkte und konvergiert nie in einem nicht steuerbaren. a = 1,0 und 1,05: alle 624 bei jeder Regel (das Gitter liegt innerhalb des steuerbaren Bereichs: Aussage über das Gitter, nicht über den Bereich). a = 1,1: Abschneiden 540, MPC N = 3: 608, ab N = 5 alle 624. **a = 1,15: 616 steuerbar, Abschneiden 410, MPC N = 20: 610 (verfehlt 6 steuerbare)**. a = 1,25: 392 steuerbar, MPC N = 20: 392, Abschneiden 276. MPC mit N = 20 konvergiert in jedem Punkt, in dem Abschneiden konvergiert (alle gemessenen a), und in keinem nicht steuerbaren | `test_b11_*` |
| B12 | Abschneiden bei sehr großen Zuständen | Am Doppelintegrator konvergiert Abschneiden auf allen 440 Punkten eines weiten Gitters (x₁ ±300, x₂ ±60, u_max = 1; R = 1, 0,1 und 0,01; 3 000 Takte) und in 500 von 500 Zufallsstarts in [−1 000, 1 000] × [−200, 200] (Zufalls-Seed 14, R = 1; Median 3 293 Takte bis Norm unter 10⁻⁶, höchstens 11 326): kein Stabilitätsunterschied in den gemessenen Fällen, nur ein Kostenunterschied. **Das Überschwingen ist groß:** die größte Norm der Bahn ist im Median das 11,4-Fache, höchstens das 93,6-Fache der Startnorm (Abschneiden). MPC (N = 10) an den ersten 40 dieser Starts (kleine Stichprobe; die Zahlen gelten für diese Ziehung und sind Größenordnungen, eine andere Ziehung gäbe andere Mediane, und sie sind nicht mit den 500 Starts des Abschneidens vergleichbar): alle 40 konvergieren, größte Norm im Median das 14,4-Fache, höchstens das 64,4-Fache; die gefahrenen Kosten liegen an diesen 40 Starts bei 16 bis 57 % der Kosten des Abschneidens | `test_b12_*` |
| B13 | Aufwand der Aktivmengen-Methode (u_max = 1, R = 1, Kaltstart bei u = 0, 10 812 QP-Aufrufe entlang der 356 Bahnen) | Iterationen im Mittel 2,37 (N = 5), 2,65 (N = 10), 2,81 (N = 20), Maximum 10 / 27 / 60; in 13,9 % der Aufrufe ist mindestens eine Grenze aktiv; in den Bahnen kommen 37 / 77 / 81 verschiedene Vorzeichenmuster der Stellfolge (−1/0/+1 je Stelle) vor. Das ist die Zahl der Muster **in den Bahnen**, nicht der Regionen eines expliziten Gesetzes. Eine Iteration ist ein Durchlauf der Hauptschleife einschließlich des letzten, der die Optimalität feststellt | `test_b13_*` |
| B14 | Das QP gegen drei andere Rechenwege | Messung an den Zufalls-QPs der Demo-Tests (andere QPs gäben andere Maxima): 400 Zufalls-QPs (N = 1 bis 7, 376 mit aktiver Grenze): eigene Methode gegen Aufzählung aller 3^N Muster größte Abweichung der Lösung 3,4·10⁻¹⁵, gegen `lsq_linear(method="bvls")` 7,9·10⁻¹⁵, gegen SLSQP 5,6·10⁻⁸. 240 MPC-QPs entlang des Wagens (N = 2 bis 7): 2,6·10⁻¹⁴ / 1,3·10⁻¹⁴ / 2,5·10⁻⁶. Für N = 10, 20, 30, 60 (Konditionszahl von H 1,2·10³ bis 9,1·10⁵, je 30 Zufallszustände) liegt `bvls` um höchstens 4,7·10⁻¹⁰ im Wert über der eigenen Lösung, SLSQP bei N = 60 um 7,1·10⁵ (versagt dort); größte KKT-Verletzung der eigenen Lösung 1,7·10⁻¹⁰. Stellfolgen-Gitter (41³ = 68 921 Folgen, N = 3): kein Gitterwert unter dem QP-Wert | `test_b14_*`, `tests/test_oracle_mpc_stellgrenzen.py` |
| B15 | Wann bricht die eigene Methode ab? | Rückfall auf `bvls` in 0 von je rund 15 000 Aufrufen beim Doppelintegrator (N = 3, 5, 10, 20, 40), 0 bei a = 1,2 für N ≤ 20 (27 000 bis 32 000 Aufrufe), **5 544 von 27 032 (20,5 %) bei a = 1,2 und N = 40** (Konditionszahl 6,5·10⁸; Band 5 bis 50 %, nie als genaue Zahl benutzt) | `test_b15_*`, `tests/test_edge_cases.py` |
| B16 | Riccati-Gewinn | K = (0,42208; 1,24393) für R = 1; geschlossene Eigenwerte 0,378 ± 0,188 i (Betrag 0,422) | `test_b16_*`, `tests/test_dynamics.py` |
| B17 | Wagen-Handbeispiel (N = 2, Q = P = I, R = 1, u_max = ½, x₀ = (7; −2)) | H = [[8, 2], [2, 4]], f = (−2; −4); unbeschränkt (0; 1); **MPC (1/8; 1/2)**, abgeschnittene unbeschränkte Folge (0; 1/2). Kosten 93 (nicht zulässig), 1495/16 = 93,4375 (MPC), 187/2 (abgeschnitten); Mehrkosten der abgeschnittenen Folge gegen MPC: 1/16 = ½·H₀₀·(1/8)² (nicht gegen die nicht zulässige unbeschränkte Lösung mit Kosten 93). Liegt u₁ an der Grenze, fehlt später Bremswirkung, und u₀ muss jetzt etwas tun (Deutung, am Beispiel gerechnet: ∂/∂u₀ = 8u₀ + 2·½ − 2 = 0). Gradient (0; −7/4): u₀ frei, u₁ oben mit negativem Gradienten. Drei Regelschritte: u₀ = 1/8, 1/2, 27/56 | `test_b17_*`, `tests/test_exact.py` |

Die Sätze unter den Experimenten der App sind reine Funktionen (`msg_evaluation.verdict_*`), geprüft an handgesetzten Werten (`tests/test_verdicts.py`) und je Schnellstart-Knopf (`tests/test_claims.py::test_preset_*`): der Skalar-Knopf meldet keinen Vorteil von MPC,
der Knopf „Zu kurzer Horizont ohne Endkosten“ meldet „teurer“.

### Abweichungen gegenüber den ersten Überschlagsrechnungen

- **Zufallsinstanzen neu gezogen.** Eine erste Rechnung mit `numpy.random` ergab für die Gegenprobe (B6) gleich in 240 / 206 / 206, billiger in 760 / 794 / 794 und größten Vorteil 69,1 / 75,3 / 80,4 %; für die großen Zufallsstarts (B12) Median 2 786 und höchstens 10 882 Takte, Überschwingen 11,2 und 90,2.
  In der Demo kommen die Zufallszahlen aus dem SplitMix64 statt aus `numpy.random` (gleicher Zufalls-Seed 14, andere Instanzen): 232 / 193 / 193, 768 / 807 / 807, 72,7 / 74,9 / 88,2 % und Median 3 293, höchstens 11 326 Takte, 11,4 und 93,6. Die Aussagen bleiben (teurer in 0 von 1 000; Abschneiden konvergiert überall; starkes Überschwingen); die Zahlen der Tabelle sind die der Demo.
- **Orakel-Abweichungen und Rückfallanteil verschoben.** Mit anderen Zufalls-QPs und relativen Toleranzen (zum Boxradius) sind die Abweichungen B14 etwas anders als in der ersten Rechnung (zum Beispiel SLSQP gegen die Methode 5,6·10⁻⁸ statt 1,0·10⁻⁷) und der Rückfall bei a = 1,2, N = 40 liegt bei 20,5 % statt 20,7 % (B15).
- **Die steuerbare Menge ist nicht immer das, was MPC N = 20 erreicht.** Eine erste Rechnung für a = 1,2 hatte MPC mit N = 20 und 40 als gleich der steuerbaren Menge gefunden. Das gilt für a = 1,2 und a = 1,25; bei a = 1,15 verfehlt MPC mit N = 20 sechs von 616 steuerbaren Punkten (B11). Die Menge N = 40 wird in der App nicht gezeigt (Horizont höchstens 20).
- **MPC bei sehr großen Zuständen gemessen (B12):** alle 40 Starts konvergieren, Überschwingen höchstens das 64,4-Fache; MPC liegt an diesen Starts bei 16 bis 57 % der Kosten des Abschneidens. Das ist eine Stichprobe von 40 Starts, nicht das ganze Gebiet; die Zahlen schwanken von Ziehung zu Ziehung.
- **Nicht gemacht:** eine zweite Rechnung für die Referenz N = 60 am Doppelintegrator (zum Beispiel dynamische Programmierung über einem Zustandsgitter). Gestützt ist sie durch B9 (Voraussage gleich gefahren ab genügend großem N) und den Skalar (B2); für den Wagen bleibt sie eine Annahme.
- **Wortgebrauch.** „Abschneiden“ steht für beide Arten (siehe Modell); „Klemmung“ bleibt der Wortgebrauch der [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo).
- **Zielwert gegen Stellfolge bei N = 40.** In einer Stichprobe von 25 Zuständen (a = 1,2, N = 40, `tests/test_edge_cases.py`) stimmen die Zielwerte (Größenordnung 10⁹) von eigener Methode und `bvls` relativ auf 10⁻⁹ überein, die Stellfolgen können in flachen Richtungen der Zielfunktion um bis zu rund 0,35 abweichen (beim Bau beobachtet, die Zahl ist nicht im Test festgeschrieben). Ein weiterer Grund für die Grenze N ≤ 20 der App.
- **Fehlstart einer ersten Rechnung** (in den Tests verankert): `lsq_linear` mit der Voreinstellung der Iterationszahl lieferte bei N = 20 / 30 / 60 einen bis zu 2,18 / 16,6 / 66,7 höheren Zielwert als die eigene Methode; mit `max_iter=100000` stimmt es auf 10⁻⁹ (`tests/test_oracle_mpc_stellgrenzen.py`). Toleranzen der Methode sind relativ zu wählen; die erste Fassung mit festem 10⁻¹² lief bei N = 60 in eine Endlosschleife.

## Quellen

Die Gleichungen sind in eigener Herleitung geschrieben und gegen die direkt summierte Kostenfunktion (Brüche) geprüft; **aus dem Rawlings/Mayne/Diehl-Buch ist nichts wörtlich übernommen** (das PDF enthält den Vermerk „Weitergabe nur mit Erlaubnis des Verlags“), nur Gleichungen und Befunde in eigenen Worten.

- Rawlings/Mayne/Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2. Aufl.: Kapitel 2 (Abschnitt 2.2, Beispiel 2.5, die Skalar-Handbeispiele): QP mit Stellgrenze, Skalar mit N = 2; die Zahl 124 für V_N⁰(10) nennt dort nach Quellenbericht die Quelle (im Original nicht gelesen), gerechnet ist 123,5 (B1); die Indizes der Quelle zählen ab 1, hier ab 0.
  Abschnitt 2.5.1 (nach Quellenbericht, im Original nicht gelesen): mit Endkosten aus der Riccati-Gleichung gilt das Gesetz u = Kx unabhängig von N (hier B4).
- Borrelli/Bemporad/Morari, *Predictive Control for Linear and Hybrid Systems*, Kapitel 11.3.1: das kondensierte QP (hier Formel im Expander 📐).
- Die Kapitel- und Seitenangaben stammen aus einer Lesenotiz (Wortlaut gelesen); **die Seitenzahlen (Rawlings S. 99 bis 101, Borrelli/Bemporad/Morari S. 218 bis 219) sind am Original noch nicht geprüft**, und es gibt keinen Test gegen einen Literaturwert (Quelle für diese Demo nicht neu eingesehen: kein Test, nur ein Eintrag „offen“).
- Die Aussage von Qin/Badgwell (nach dem Quellenbericht: bei starken Beschränkungen sei unbeschränktes LQR mit Anti-Windup deutlich schlechter als MPC) wird hier **nicht** als Beleg benutzt; die gemessene Spanne ist 3,19 % bis 146,37 % je nach Verhältnis Zustand zu Grenze (B5).
- Offen: die Frage, wie „LQR, dann abschneiden“ mit einem PI-Regler und Back-Calculation abschneidet, steht in [`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo).

## Ehrliche Grenzen

- **Nur Boxgrenzen auf u.** Zustandsgrenzen, Zulässigkeit und Terminalmenge fehlen; mit Zustandsgrenzen kann das QP unzulässig werden, hier ist es immer lösbar ([`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo)).
- **Ein Modell, ein Q.** Der Wagen mit a ∈ {1; 1,05; 1,1; 1,15; 1,2; 1,25}, der Skalar, Q = I, R ∈ {0,1; 1; 10}; keine Strecke mit mehreren Eingängen, kein Doppelintegrator mit Reibung, Gitter statt Rand des Anziehungsgebiets.
- **„Nie teurer“ ist keine Garantie**, sondern eine Zählung (B5: 3 × 356 Startzustände, B6: 1 000 Zufallsinstanzen) und gilt mit Endkosten DARE; mit Endkosten nur Q ist MPC bei N = 2 in 322 von 356 Startzuständen teurer (B10). Terminalkosten P sind eine Untergrenze für den Rest, keine Obergrenze.
- **Die Referenz N = 60** ist kein bewiesener unendlicher Horizont (siehe oben).
- **Kein Rauschen, keine Störung, kein Verzug, kein Modellfehler.** Mit Stellgrenze und Störung oder Verzug ist nichts gemessen ([`plan-stoerung-demo`](https://github.com/sebastian-hanisch/plan-stoerung-demo), [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo), [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo)).
- **Horizont höchstens 20.** Darüber wird H schlecht konditioniert: bei a = 1,2 und N = 40 bricht die Methode in 20,5 % der Aufrufe ab (B15). Die dünne Form (Zustände als Variablen) wäre ein Ausweg und ist nicht umgesetzt.
- **Rechenzeit ist nicht gemessen und nicht festgeschrieben;** Warmstart und der Vergleich mit Innere-Punkte-Verfahren oder ADMM kommen nicht vor (Thema Konvexe und konische Optimierung). Die Zahl der Iterationen (B13) ist ein Aufwandsmaß, kein Zeitmaß.
- **Gleichstände an Schwellen:** „gleich“ heißt relativ höchstens 10⁻⁹; zwischen 10⁻¹² und 10⁻⁶ liegt in den Messungen kein Punkt. Zählungen aus der Neu-Rechnung gelten unter Linux als Band (± 2).
- **Warum im Skalar und bei N = 1 kein Unterschied entsteht und warum MPC mit Endkosten DARE nie teurer war, ist nicht erklärt**, nur gemessen. Der Satz „im Skalar gilt immer“ steht nirgends.

## Verifikation

- **Kern:** das QP gegen Handrechnung (`tests/test_qp.py`, `tests/test_exact.py`: Skalar, Wagen N = 2 in Brüchen), gegen die Aufzählung aller 3^N Muster, `bvls` und SLSQP und ein Stellfolgen-Gitter (**Orakel**, `tests/test_oracle_mpc_stellgrenzen.py`); Riccati-Gewinn gegen die Riccati-Rekursion; kondensierte Kosten gegen die direkt summierte Kostenfunktion; gefahrene Kosten im Skalar gegen exakte Brüche und dynamische Programmierung; steuerbare Menge (lineares Programm) gegen beschränkte kleinste Quadrate.
- **Eigenschaften** (`tests/test_eigenschaften_mpc_stellgrenzen.py`): Symmetrie, Skalierung, Kosten nie unter x'P x, größere Grenze senkt den QP-Wert nie, u_max → ∞ liefert −K x, N = 1 gleich Abschneiden für beliebiges P, positiver Zielfaktor ändert nichts, Gegenprobe „nie teurer“.
- **Claims:** jede Zahl der README (Tabelle oben, `tests/test_claims.py`); **Randfälle** (`tests/test_edge_cases.py`): u_max 10⁻⁶ und 10⁶, x₀ = 0, |x| = 5/3 und 3, N = 1 und 2, R → 0, a = 1,2 mit N = 40, entartete Grenze, Permalink-Werte außerhalb der Regler.
- **Verdikte:** reine Funktionen, an handgesetzten Werten (`tests/test_verdicts.py`); **Schnellstart-Knöpfe:** vollständig, eindeutig, der erste gleich den Standardeinstellungen (`tests/test_presets.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Knopf, jeder Regler am Minimum und Maximum), Footer, Layout (Schnellstart oben in der Seitenleiste, fester Kopf, jeder Regler steht in `SETTING_SPECS` und seine Änderung hebt die Markierung auf), Laufzeit-Importe.
- **Zufall:** `tests/test_rng.py` (erste Werte des SplitMix64 bei Zufalls-Seed 0).
- **Nicht getestet:** Browserdarstellung der Karten und Diagramme (nur AppTest), echte Mausklicks.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, drei Kernfragen, sechs Experimente, Grenzen, Formeln |
| `msg_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Schnellstart-Knöpfe |
| `msg_qp.py` | kondensiertes QP, Aktivmengen-Methode (mit Spur), Rückfall |
| `msg_dynamics.py` | Strecken, Riccati-Gewinn, beide Regeln, Regelkreis, Gitter, Zählung, steuerbare Menge |
| `msg_exact.py` | exakte Handrechnung mit Brüchen (Skalar, Wagen N = 2) |
| `msg_evaluation.py` | Einstellungen, Rechnungen der Experimente, Verdikte, Tabelle der Grenzen |
| `msg_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `msg_presets.py` | Permalink (Begrenzen, Einrasten), Schnellstart-Knöpfe, Hervorhebung des aktiven Knopfes |
| `msg_layout.py` | Fester Kopf (Stylesheet) |
| `msg_rng.py` | SplitMix64 (unveränderte Kopie der Vorlage des Website-Repos; nur Tests und `tools/`) |
| `data/msg_frozen_*.json` | vorgerechnete Messreihen (8 Dateien) |
| `tools/make_frozen.py` | erzeugt `data/` neu (`make_frozen.py <teil>`; Laufzeit je Teil Sekunden bis wenige Minuten) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Claims, Randfälle, Verdikte, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe, Zufallsgenerator |

## Bewusst nicht umgesetzt

- **Zustandsgrenzen, Terminalmenge, rekursive Zulässigkeit** (eigene Demo: [`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo)).
- **ADMM, Warmstart, Innere-Punkte-Vergleich, explizites MPC** (Regionen eines expliziten Gesetzes werden nicht gezählt; die 37 / 77 / 81 Muster in B13 sind eine andere Zahl).
- **Zufall in der App** (Gitter statt Stichproben; sonst hieße der Regler „Zufalls-Seed“).
- **PDF-Export.** Die Ergebnisse sind Zählungen mit ihren Bedingungen, die in README und App stehen.
- **Ein PI-Regler mit Anti-Windup** (Gegenfrage in [`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo)).
- **Der Horizont N = 40 in der App** (siehe Grenzen).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Suite hat 183 Tests (Laufzeit auf dem Entwicklungsrechner bei einem Lauf rund 104 Sekunden, nicht festgeschrieben).

Die vorgerechneten Messreihen lassen sich neu erzeugen: `python tools/make_frozen.py kurz d1 skalar gegenprobe weit instabil orakel rueckfall` (zusammen einige Minuten).

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regelung von LQR bis MPC: vom Plan zur Regel](https://sebastianhanisch.net/konzepte-regelung.html).
