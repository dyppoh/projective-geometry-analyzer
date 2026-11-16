<!-- Erklärung auf der Startseite: Ablauf, Regeln, Urteil, Begriffe -->

<section class="info">
	<h2 class="label">Wie es funktioniert</h2>
	<ol class="steps">
		<li class="panel">
			<b class="mono">01</b>
			<h3>Bild wählen</h3>
			<p>Du kannst ein eigenes Bild hochladen oder eines der Testbilder verwenden.</p>
			<p>Alternativ kannst du ganze Projekte aus Bild, Linien und Gruppen als JSON exportieren und importieren.</p>
		</li>
		<li class="panel">
			<b class="mono">02</b>
			<h3>Fluchtlinien finden und gruppieren</h3>
			<p>
				Im Bild müssen Linien eingezeichnet werden, welche parallel, auf parallelen Ebenen oder senkrecht zueinander verlaufen.
			</p>
			<p>
				Es kann automatisch durchgeführt werden, das System macht allerdings grobe Fehler.
				<br>Alternativ kannst du ein MLLM deiner Wahl übernehmen lassen, welches HTTP-Anfragen durchführen kann.
			</p>
			<p>
				Allgemein gilt: du kannst und solltest die Linien und Gruppen unbedingt kontrollieren.
				Denn die Aussagekraft der folgenden Auswertung ist vollkommen von der Korrektheit der Linien abhängig.
			</p>
		</li>

		<li class="panel">
			<b class="mono">03</b>
			<h3>Auswerten</h3>
			<p>Die gruppierten Linien müssen in einem Bild bestimmte Regeln einhalten. Um das zu prüfen, werden ein Grundwert und Abstandsmaße berechnet.</p>
			<p>Der Grundwert wird aus allen Linien bestimmt und soll beschreiben, wie genau die Messungen allgemein sind.</p>
			<p>Wenn bestimmte Abstandsmaße wesentlich größer als der Grundwert sind, werden sie als Befund erkannt. Enthält ein Bild einen gewissen Anteil an Befunden, könnte das Bild unecht sein.</p>
		</li>
	</ol>
</section>

<section class="info">
	<h2 class="label">Die Regeln</h2>
	<div class="rules">
		<article class="panel">
			<!-- <img src="" alt="Veranschaulichung von Regel 1 - "> -->
			<span class="tag mono">Regel 1</span>
			<h3>Fluchtpunkte</h3>
			<p>
				Wenn Linien innerhalb eines Raumes parallel fliehen, treffen sie sich in einem Bild in einem Fluchtpunkt.
			</p>
			<p class="note">Der Median-Abstand aller Linien ergibt den Grundwert.</p>
		</article>

		<article class="panel">
			<!-- <img src="" alt="Veranschaulichung von Regel 2 - "> -->
			<span class="tag mono">Regel 2</span>
			<h3>Fluchtpunkte nach Ebenen</h3>
			<p>
				Wenn parallel fliehende Linien nach Ebenen getrennt werden, müssen sie jeweils den gleichen Fluchtpunkt ergeben. 
			</p>
		</article>

		<article class="panel">
			<!-- <img src="" alt="Veranschaulichung von Regel 3 und 4 - "> -->
			<span class="tag mono">Regel 3 und 4</span>
			<h3>Fluchthorizont</h3>
			<p>
				Für alle fliehenden Linien, die auf gleichen oder parallelen Ebenen verlaufen, liegt ihr jeweiliger Fluchtpunkt auf einem Fluchthorizont.
			</p>
		</article>

		<article class="panel">
			<!-- <img src="" alt="Veranschaulichung von Regel 5 - "> -->
			<span class="tag mono">Regel 5</span>
			<h3>Hauptpunkt eines Bildes</h3>
			<p>
				Für drei fliehende Linien, die wie Koordinatenachsen alle senkrecht zueinander verlaufen, ergibt sich aus ihren Fluchtpunkten ein Dreieck.
				Das Orthozentrum dieses Dreiecks ist immer der gleiche Hauptpunkt des Bildes.
			</p>
		</article>
	</div>
</section>

<div class="duo">
	<section class="info">
		<h2 class="label"> Die Auswertung</h2>
		<div class="panel text">
			<p>
				Der Grundwert ist der Medianabstand aller imperfekten Fluchtpunkte nach Regel 1. Sie wird somit nicht beurteilt.
			</p>
			<p>
				Für die Regeln 2 – 5 gilt jeweils eine Grenze von einem gewissen Faktor über den Grundwert. 
				Übertritt der Median der Abstände den definierten Schwellenwert, wird dies als Befund erkannt.
			</p>
			<ul class="verdicts">
				<li><i class="ok"></i><span><strong>Annahme: echt</strong> – da keine Befunde identifiziert werden konnten</span></li>
				<li><i class="bad"></i><span><strong>unecht</strong> – sobald mindestens ein Befund vorliegt</span></li>
				<li>
					<i class="na"></i><span><strong>nicht bewertbar</strong> – Es muss min. eine Regel auswertbar sein</span>
				</li>
			</ul>
		</div>
	</section>

	<section class="info">
		<h2 class="label">Begriffe</h2>
		<dl class="panel text terms">
			<dt>Parallele Linien</dt>
			<dd>Linien, die in dem ursprünglichen 3D-Raum des Bildes parallel, also genau gleichgerichtet verlaufen. Sie haben wie Bahnschienen an jeder Stelle den gleichen Abstand.</dd>
			<dt>Ebenen</dt>
			<dd>Eine flache, zweidimensionale Fläche wie ein Fußboden.</dd>
			<dt>Parallele Ebenen</dt>
			<dd>Zwei oder mehr flache Ebenen, die linke und rechte Seite eines Würfels exakt gleichgerichtet sind.</dd>
			<dt>Senkrechte Ebenen</dt>
			<dd>Drei flache Ebenen, die wie die linke, obere und vordere Seite eines Würfel alle jeweils senkrecht zueinander verlaufen.</dd>
		</dl>
	</section>
</div>

<style>
	.info {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	h3 {
		font-size: 15px;
		margin-bottom: 0.4rem;
	}
	p {
		margin: 0;
		font-size: 14px;
		line-height: 1.5;
	}
	p + p {
		margin-top: 0.6rem;
	}

	/* Ablauf */
	.steps {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(200px, 100%), 1fr));
		gap: 1rem;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.steps li {
		padding: 1rem 1.1rem 1.1rem;
	}
	.steps b {
		display: block;
		margin-bottom: 0.6rem;
		font-size: 12px;
		font-weight: 600;
		color: var(--accent);
	}
	.steps p {
		color: var(--muted);
	}
	.steps strong {
		color: var(--ink);
		font-weight: 600;
	}

	/* Regeln */
	.rules {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(230px, 100%), 1fr));
		gap: 1rem;
	}
	.rules article {
		display: flex;
		flex-direction: column;
		padding: 1rem 1.1rem 1.1rem;
	}
	svg {
		display: block;
		width: 100%;
		height: 72px;
		margin-bottom: 0.9rem;
		padding: 6px;
		background: var(--paper);
		border: 1px solid var(--hair);
		overflow: visible;
	}
	svg line,
	svg polygon {
		stroke-linecap: round;
		fill: none;
	}
	.ln line {
		stroke: var(--ink);
		stroke-width: 2;
	}
	.ext line {
		stroke: var(--muted);
		stroke-width: 1;
		stroke-dasharray: 3 3;
	}
	.hz {
		stroke: var(--accent);
		stroke-width: 1.5;
		stroke-dasharray: 5 3;
	}
	.tri {
		stroke: var(--ink);
		stroke-width: 1.5;
	}
	.alt line {
		stroke: var(--warn);
	}
	.vp {
		fill: var(--accent);
	}
	.vp.alt {
		fill: var(--warn);
	}
	.dot {
		fill: var(--ink);
	}
	.tag {
		align-self: flex-start;
		margin-bottom: 0.5rem;
		padding: 0.2em 0.5em;
		background: var(--ink);
		color: var(--paper);
		font-size: 11px;
		letter-spacing: 0.08em;
	}
	.note {
		margin-top: auto;
		padding-top: 0.7rem;
		color: var(--muted);
		font-size: 13px;
	}

	/* Urteil & Begriffe */
	.duo {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr));
		gap: 1rem 2rem;
	}
	.text {
		flex: 1;
		margin: 0;
		padding: 1.1rem 1.2rem;
	}
	.verdicts {
		display: grid;
		gap: 0.45rem;
		margin: 0.9rem 0 0;
		padding: 0.9rem 0 0;
		border-top: 1px solid var(--hair);
		list-style: none;
		font-size: 14px;
	}
	.verdicts li {
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.verdicts i {
		flex: none;
		width: 9px;
		height: 9px;
		transform: rotate(45deg);
	}
	.ok {
		background: var(--ok);
	}
	.bad {
		background: var(--bad);
	}
	.na {
		background: var(--hair);
	}
	.terms {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: 0.7rem 1.2rem;
		font-size: 14px;
		line-height: 1.45;
	}
	dt {
		font: 600 12px/1.6 var(--mono);
		letter-spacing: 0.04em;
	}
	dd {
		margin: 0;
		color: var(--muted);
	}
	@media (max-width: 760px) {
		.steps,
		.rules {
			gap: 0.75rem;
		}
		/* Schritte kompakt: Nummer links, Text rechts */
		.steps li {
			display: grid;
			grid-template-columns: 2rem minmax(0, 1fr);
			padding: 0.85rem 1rem;
		}
		.steps b {
			grid-row: span 2;
			margin: 0;
			line-height: 1.5;
		}
		.rules article,
		.text {
			padding: 0.9rem 1rem 1rem;
		}
		svg {
			height: 64px;
		}
		.duo {
			gap: 2rem;
		}
	}
	@media (max-width: 520px) {
		.terms {
			grid-template-columns: 1fr;
			gap: 0.2rem;
		}
		dd {
			margin-bottom: 0.6rem;
		}
	}
</style>
