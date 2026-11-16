<script lang="ts">
	import Canvas from './Canvas.svelte';
	import { extend, pct } from './geometry';
	import type { CheckGroup, Finding, Kriterium, Result } from './types';

	interface Props {
		src: string;
		result: Result;
		/** Linien/Gruppen wurden seit dieser Auswertung geändert */
		stale: boolean;
	}

	let { src, result, stale }: Props = $props();

	const RULES: Record<string, string> = {
		'1': 'Fluchtpunkt der Richtung',
		'2': 'Fluchtpunkt der Ebene',
		'2-cross': 'Fluchtpunkte der Ebenen',
		'3/4': 'Fluchthorizont',
		'5': 'Orthozentren'
	};
	const STATUS = {
		befund: 'Befund',
		ok: 'unauffällig',
		grundwert: 'Grundwert (nicht bewertet)',
		ungeprueft: 'nicht prüfbar',
		entfallen: 'entfallen'
	};
	const WERT = { median: 'Median-Abstand' };

	let canvas = $state<Canvas>();
	let scale = $state(1);
	let selectedId = $state<string | null>(null);

	const v = $derived(result.verdict);
	const selected = $derived(result.checks.find((c) => c.id === selectedId) ?? null);
	const finding = $derived(v.findings.find((f) => f.group === selectedId) ?? null);
	const far = $derived(Math.hypot(result.imageWidth, result.imageHeight) * 20);
	const legend = $derived([...new Map((selected?.layers ?? []).map((l) => [l.label, l.lineColor])).entries()]);
	/** Linien über der Grenze (Rule 2: Verweise sind Linien-IDs) */
	const overLines = $derived.by(() => {
		const ids = new Set(finding?.kriterien.flatMap((k) => k.over.map((o) => o.ref)) ?? []);
		return result.lines.filter((l) => ids.has(l.id));
	});

	function show(check: CheckGroup) {
		if (selectedId === check.id) {
			selectedId = null;
			canvas?.fit();
			return;
		}
		selectedId = check.id;
		// Ausschnitt: Bild + Fluchtpunkte (begrenzt, sehr ferne Punkte bleiben draußen)
		const W = result.imageWidth;
		const H = result.imageHeight;
		const pts = check.layers
			.flatMap((l) => l.points ?? [])
			.filter((p) => Math.abs(p.x - W / 2) < 4 * W && Math.abs(p.y - H / 2) < 4 * H);
		const xs = [0, W, ...pts.map((p) => p.x)];
		const ys = [0, H, ...pts.map((p) => p.y)];
		const x = Math.min(...xs);
		const y = Math.min(...ys);
		canvas?.fit({ x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y });
	}

	function showFinding(f: Finding) {
		const c = result.checks.find((c) => c.id === f.group);
		if (c) show(c);
	}

	/** Wert als Vielfaches des Grundwerts (Grenze = Faktor × Grundwert) */
	function inGw(k: Kriterium, value = k.value) {
		const r = (value * k.faktor) / k.threshold;
		return `${r < 100 ? r.toFixed(1).replace('.', ',') : r.toFixed(0)} × Grundwert`;
	}

	const entfallen = $derived(result.checks.filter((c) => c.status === 'entfallen').length);

	function lines(text: string) {
		return text
			.split('\n')
			.filter((l) => l.trim())
			.map((l) => ({ item: l.startsWith('- '), text: l.replace(/^- /, '') }));
	}
</script>

<div class="view">
	<div class="canvas">
		<Canvas bind:this={canvas} bind:scale {src} width={result.imageWidth} height={result.imageHeight} dim={!!selected}>
			{#each selected?.layers ?? [] as layer}
				{#each layer.lines ?? [] as l}
					{@const [a, b] = layer.extend === false ? [l.point1, l.point2] : extend(l.point1, l.point2, far)}
					<line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={layer.lineColor} />
				{/each}
				{#each layer.points ?? [] as p}
					<rect
						class="point"
						x={p.x - 6 * scale}
						y={p.y - 6 * scale}
						width={12 * scale}
						height={12 * scale}
						fill={layer.pointColor}
						transform="rotate(45 {p.x} {p.y})"
					/>
				{/each}
			{/each}
			{#each overLines as l (l.id)}
				<line class="over" x1={l.point1.x} y1={l.point1.y} x2={l.point2.x} y2={l.point2.y} />
			{/each}
		</Canvas>
		{#if selected}
			<div class="legend">
				<strong>{selected.title}</strong>
				{#each legend as [label, color]}
					<span><i style:background={color}></i>{label}</span>
				{/each}
				{#if overLines.length}
					<span><i style:background="var(--bad)"></i>Linien über der Grenze</span>
				{/if}
				{#if selected.reason}
					<p class="why">{selected.reason}</p>
				{/if}
			</div>
		{/if}
	</div>

	<aside>

		<section class="verdict {v.decision.replace(' ', '-')}">
			<h3 class="label"> Urteil</h3>
			<div class="decision">{v.decision === 'echt' ? 'Annahme: echt' : v.decision}</div>
			{#if v.decision === 'echt'}
				<p class="reason">{v.reason}</p>
			{/if}
			<dl class="facts mono">
				<dt>Befunde</dt>
				<dd>{v.befunde} von {v.kriterien} Kriterien</dd>
				<dt>Prüfgruppen</dt>
				<dd>{v.pruefgruppen}</dd>
				{#if v.grundwert}
					<dt>Grundwert</dt>
					<dd>
						{v.grundwert.value === null ? 'nicht bestimmbar' : pct(v.grundwert.value)}
						<small>(mit {v.grundwert.n} Fluchtlinien bestimmt)</small>
					</dd>
				{/if}
			</dl>

			{#if v.explanation}
				<details>
					<summary>Begründung im Detail</summary>
					<div class="explanation">
						{#each lines(v.explanation) as l}
							<p class:item={l.item}>{l.text}</p>
						{/each}
					</div>
				</details>
			{/if}
		</section>

		<section>
			<h3 class="label"> Argumente <span class="mono">{v.findings.length}</span></h3>
			{#each v.findings as f}
				<button class="row finding" class:on={selectedId === f.group} onclick={() => showFinding(f)}>
					<span class="rule mono">R{f.rule.replace('-cross', '')}</span>
					<span class="text">
						<span>{f.subject}</span>
						<small>{f.title}</small>
					</span>
					<span class="crit mono">
						{#each f.kriterien as k}
							<span
								class:hit={k.befund}
								title="Median {pct(k.value)} = {inGw(k)} · Grenze {pct(k.threshold)} = {String(k.faktor).replace(
									'.',
									','
								)} × Grundwert"
							>
								{WERT[k.wert]}
								{pct(k.value)}
							</span>
						{/each}
					</span>
				</button>
			{:else}
				<p class="none">Keine Befunde.</p>
			{/each}
		</section>

		<section>
			<h3 class="label">
				Alle Prüfungen <span class="mono">{result.checks.length - entfallen} + {entfallen} entfallen</span>
			</h3>
			{#each result.checks as c (c.id)}
				<button class="row" class:on={selectedId === c.id} class:skipped={c.status === 'entfallen'} onclick={() => show(c)}>
					<span class="sev {c.status}" title={STATUS[c.status]}></span>
					<span class="rule mono">R{c.rule.replace('-cross', '')}</span>
					<span class="text">
						<span>{c.subject}</span>
						<small>{c.title === 'Gruppe nicht verwendet' ? c.title : (RULES[c.rule] ?? '')} · {STATUS[c.status]}</small>
						{#if c.reason && c.status !== 'grundwert'}<small class="grund">{c.reason}</small>{/if}
					</span>
				</button>
			{/each}
		</section>
	</aside>
</div>

<style>
	.view {
		display: grid;
		grid-template-columns: 1fr 420px;
		height: 100%;
		min-height: 0;
	}
	.canvas {
		position: relative;
		min-height: 0;
	}
	line {
		stroke-width: 1.5;
		vector-effect: non-scaling-stroke;
		opacity: 0.9;
	}
	line.over {
		stroke: var(--bad);
		stroke-width: 4;
		opacity: 1;
	}
	.point {
		stroke: white;
		stroke-width: 1.5;
		vector-effect: non-scaling-stroke;
	}
	.legend {
		position: absolute;
		left: 12px;
		top: 12px;
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		max-width: 360px;
		padding: 0.7rem 0.9rem;
		background: rgba(20, 23, 28, 0.85);
		color: white;
		font-size: 13px;
		pointer-events: none;
	}
	.legend i {
		display: inline-block;
		width: 14px;
		height: 3px;
		margin-right: 0.5rem;
		vertical-align: middle;
	}
	.legend em {
		color: rgba(255, 255, 255, 0.55);
		font-size: 12px;
	}

	aside {
		display: flex;
		flex-direction: column;
		gap: 1.6rem;
		padding: 1.25rem;
		border-left: 1px solid var(--hair);
		background: var(--panel);
		overflow-y: auto;
	}
	.stale {
		padding: 0.6rem 0.8rem;
		border-left: 4px solid var(--warn);
		background: rgba(201, 154, 0, 0.1);
		font-size: 13px;
	}

	.verdict {
		--c: var(--muted);
	}
	.verdict.echt {
		--c: var(--ok);
	}
	.verdict.unecht {
		--c: var(--bad);
	}
	.decision {
		display: inline-block;
		padding: 0.25em 0.8em 0.2em;
		background: var(--c);
		color: white;
		font-size: 30px;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
	}
	.reason {
		margin: 0.7rem 0 0;
		font-weight: 600;
	}

	.facts {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 0.25rem 1rem;
		margin: 0.9rem 0 0;
		font-size: 12px;
	}
	.facts dt {
		color: var(--muted);
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}
	.facts dd {
		margin: 0;
	}

	summary {
		margin-top: 0.9rem;
		font-size: 13px;
		cursor: pointer;
		list-style: none;
	}
	summary::before {
		content: '▸ ';
		color: var(--c);
	}
	details[open] summary::before {
		content: '▾ ';
	}
	.explanation {
		margin-top: 0.6rem;
		font-size: 13px;
		color: var(--muted);
	}
	.explanation p {
		margin: 0.2rem 0;
	}
	.explanation .item {
		padding-left: 1rem;
		position: relative;
	}
	.explanation .item::before {
		content: '';
		position: absolute;
		left: 0.2rem;
		top: 0.5em;
		width: 5px;
		height: 5px;
		background: var(--c);
	}

	.row {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		width: 100%;
		padding: 0.5rem 0.4rem;
		border: 0;
		border-bottom: 1px solid var(--hair);
		background: transparent;
		text-align: left;
	}
	.row:hover {
		background: rgba(20, 23, 28, 0.04);
	}
	.row.on {
		background: var(--ink);
		color: var(--paper);
	}
	.rule {
		flex: none;
		width: 2.6em;
		padding: 0.15em 0;
		border: 1px solid currentColor;
		text-align: center;
		font-size: 11px;
	}
	.finding .rule {
		border-color: var(--bad);
		color: var(--bad);
	}
	.text {
		flex: 1;
		display: flex;
		flex-direction: column;
		font-size: 14px;
		min-width: 0;
	}
	small {
		font-size: 12px;
		opacity: 0.6;
	}
	.crit {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 0.2rem;
		font-size: 11px;
	}
	.crit span {
		padding: 0.05em 0.4em;
		border: 1px solid var(--hair);
		opacity: 0.55;
		white-space: nowrap;
	}
	.crit span.hit {
		border-color: var(--bad);
		background: var(--bad);
		color: white;
		opacity: 1;
	}
	.sev {
		flex: none;
		width: 10px;
		height: 10px;
		background: var(--hair);
	}
	.sev.ok {
		background: var(--ok);
	}
	.sev.befund {
		background: var(--bad);
	}
	.sev.entfallen {
		background: repeating-linear-gradient(45deg, currentColor 0 1.5px, transparent 1.5px 4px);
		border: 1px solid currentColor;
		opacity: 0.6;
	}
	.sev.ungeprueft {
		border: 1px solid currentColor;
		background: transparent;
	}
	.row.skipped .text > span:first-child {
		color: var(--muted);
	}
	.row.on.skipped .text > span:first-child {
		color: inherit;
	}
	.grund {
		margin-top: 0.15rem;
		font-style: italic;
	}
	.why {
		margin: 0.3rem 0 0;
		color: rgba(255, 255, 255, 0.8);
	}
	.sev.grundwert {
		background: transparent;
		border: 2px solid currentColor;
		transform: rotate(45deg);
	}
	.none {
		margin: 0;
		color: var(--muted);
		font-size: 14px;
	}
</style>
