<script lang="ts">
	import { tick } from 'svelte';
	import Canvas from './Canvas.svelte';
	import { PALETTE, PLANE_COLORS, distToSegment, uniqueId } from './geometry';
	import { DETECTORS, type Detector, type Ebene, type EbenenGruppe, type Line, type Point, type Richtung, type Scene } from './types';

	interface Props {
		src: string;
		scene: Scene;
		/** seit der letzten Auswertung geändert */
		dirty: boolean;
		busy: '' | 'annotate' | 'mllm' | 'evaluate';
		/** Linienerkennung für automatisch / MLLM */
		detector: Detector;
		/** Linien erkennen und gruppieren lassen: M-LSD bzw. Hough + TLC (ersetzt alles) */
		onannotate: () => void;
		/** Linien erkennen (M-LSD bzw. Hough), gruppieren lässt ein MLLM über /group/<token> */
		onmllm: () => void;
		onevaluate: () => void;
	}

	let {
		src,
		scene = $bindable(),
		dirty = $bindable(),
		detector = $bindable(),
		busy,
		onannotate,
		onmllm,
		onevaluate
	}: Props = $props();
	const detectorName = $derived(DETECTORS[detector].name);

	type Tool = 'select' | 'assign' | 'draw';
	const TOOLS: { id: Tool; label: string; hint: string }[] = [
		{ id: 'select', label: 'Auswählen', hint: 'Linie anklicken, Endpunkte ziehen, Entf löscht.' },
		{ id: 'assign', label: 'Zuordnen', hint: 'Klick nimmt eine Linie in die aktive Ebene auf oder heraus.' },
		{ id: 'draw', label: 'Zeichnen', hint: 'Erster Klick setzt den Anfang, zweiter das Ende. Esc bricht ab.' }
	];
	type ListKey = 'horizonte' | 'senkrecht';
	const LISTS: { key: ListKey; title: string; rule: string; hint: string; prefix: string; item: string }[] = [
		{
			key: 'horizonte',
			title: 'Gleiche und Parallele Ebenen',
			rule: 'Rule 3/4',
			hint: 'Gruppiere hier gleiche oder parallele Ebenen wie zwei gegenüberliegende Seiten eines Würfels.',
			prefix: 'e',
			item: 'Ebenen-Gruppe'
		},
		{
			key: 'senkrecht',
			title: 'Senkrechte Ebenen',
			rule: 'Rule 5',
			hint: 'Gruppe hier drei zueinander senkrechte Richtungen wie die Kanten eines Würfels.',
			prefix: 's',
			item: 'Senkrecht'
		}
	];
	const POOL = '#b8bcc2';
	const CONFLICT = '#ff2e2e';

	let tool = $state<Tool>('select');
	/** "Manuell annotieren" gewählt - Startauswahl ausblenden */
	let manual = $state(false);
	/** aktive Ebene: Ziel für Zeichnen und Zuordnen */
	let activeId = $state<string | null>(null);
	/** in der Seitenleiste gewählte Richtung bzw. Gruppe (Hervorhebung, Umbenennen) */
	let focus = $state<{ kind: 'richtung' | ListKey; id: string } | null>(null);
	let selectedId = $state<string | null>(null);
	/** aufgeklappte Ebenen (Linien sichtbar) */
	let expanded = $state(new Set<string>());
	let onlyFocus = $state(false);
	/** Zeichnen: Anfangspunkt nach dem ersten Klick, Mausposition für die Vorschau */
	let start = $state<Point | null>(null);
	let cursor = $state<Point | null>(null);
	let drag: { line: Line; end: 'point1' | 'point2' } | null = null;
	let scale = $state(1);
	let dropTarget = $state<string | null>(null);
	let tree = $state<HTMLElement>();

	// ---------- Nachschlagen ----------
	const info = $derived(
		new Map(scene.richtungen.flatMap((r, ri) => r.ebenen.map((e, ei) => [e.id, { e, r, ri, ei }] as const)))
	);
	const lineById = $derived(new Map(scene.lines.map((l) => [l.id, l])));
	/** Linie -> IDs der Ebenen, in denen sie steht */
	const membership = $derived.by(() => {
		const m = new Map<string, string[]>();
		for (const r of scene.richtungen)
			for (const e of r.ebenen) for (const id of e.lineIds) m.set(id, [...(m.get(id) ?? []), e.id]);
		return m;
	});
	const active = $derived(activeId ? (info.get(activeId) ?? null) : null);
	const selected = $derived(lineById.get(selectedId ?? '') ?? null);
	const focusGroup = $derived(
		focus && focus.kind !== 'richtung' ? (scene[focus.kind].find((g) => g.id === focus!.id) ?? null) : null
	);
	const focusRichtung = $derived(
		(focus?.kind === 'richtung' ? scene.richtungen.find((r) => r.id === focus!.id) : undefined) ??
			active?.r ??
			(selected ? info.get(membership.get(selected.id)?.[0] ?? '')?.r : undefined) ??
			null
	);
	/** Ziel für neue Linien: aktive Ebene, sonst die erste Ebene der gewählten Richtung */
	const target = $derived(
		active?.e ?? (focus?.kind === 'richtung' ? (focusRichtung?.ebenen[0] ?? null) : null)
	);
	/** hervorgehobene Linien: gewählte Richtung/Gruppe, sonst aktive Ebene */
	const highlight = $derived.by(() => {
		if (focus?.kind === 'richtung' && focusRichtung) return new Set(focusRichtung.ebenen.flatMap((e) => e.lineIds));
		if (focusGroup) return new Set(focusGroup.ebenen.flatMap((id) => info.get(id)?.e.lineIds ?? []));
		if (active) return new Set(active.e.lineIds);
		return null;
	});
	const unassigned = $derived(scene.lines.filter((l) => !membership.has(l.id)).length);

	function label(id: string): string {
		const i = info.get(id);
		return i ? `${i.r.name} · ${i.e.name}` : id;
	}
	function dirColor(ri: number): string {
		return PALETTE[ri % PALETTE.length];
	}
	/** Kürzel der Gruppen, in denen eine Ebene steht (H1, S2, ...) */
	function badges(ebeneId: string): string[] {
		return LISTS.flatMap((L) =>
			scene[L.key].flatMap((g, k) => (g.ebenen.includes(ebeneId) ? [`${L.prefix.toUpperCase()}${k + 1}`] : []))
		);
	}
	function length(id: string): number {
		const l = lineById.get(id);
		return l ? Math.round(Math.hypot(l.point2.x - l.point1.x, l.point2.y - l.point1.y)) : 0;
	}

	// ---------- Darstellung ----------
	const drawn = $derived(
		scene.lines.map((l) => {
			const ebenen = (membership.get(l.id) ?? []).map((id) => info.get(id)!);
			const dirs = new Set(ebenen.map((x) => x.r.id));
			const hl = highlight?.has(l.id) ?? false;
			let color = POOL;
			let dash: string | undefined = '4 3';
			if (dirs.size > 1) {
				[color, dash] = [CONFLICT, '8 4'];
			} else if (ebenen.length) {
				dash = undefined;
				const inFocus = focusRichtung && ebenen[0].r.id === focusRichtung.id;
				color = inFocus ? PLANE_COLORS[ebenen[0].ei % PLANE_COLORS.length] : dirColor(ebenen[0].ri);
			}
			return {
				l,
				color,
				dash,
				edge: ebenen.length > 1 && dirs.size === 1,
				width: hl ? 3.5 : 2,
				opacity: highlight && !hl ? 0.22 : 0.95,
				hidden: onlyFocus && highlight !== null && !hl
			};
		})
	);

	// ---------- Linien ----------
	function changed() {
		dirty = true;
	}

	function hit(p: Point): Line | null {
		let best: Line | null = null;
		let bestD = 8 * scale;
		for (const d of drawn) {
			if (d.hidden) continue;
			const dist = distToSegment(p, d.l.point1, d.l.point2);
			if (dist < bestD) [best, bestD] = [d.l, dist];
		}
		return best;
	}

	function ondown(p: Point): boolean {
		if (tool === 'select' && selected) {
			for (const end of ['point1', 'point2'] as const) {
				if (Math.hypot(selected[end].x - p.x, selected[end].y - p.y) < 10 * scale) {
					drag = { line: selected, end };
					return true;
				}
			}
		}
		return false;
	}

	function onmove(p: Point) {
		if (drag) {
			drag.line[drag.end] = p; // eine Linie - alle Ebenen sehen die Änderung
			changed();
		}
	}

	function onup() {
		drag = null;
	}

	function onclick(p: Point) {
		if (tool === 'draw') {
			if (!start) {
				start = p;
			} else if (Math.hypot(p.x - start.x, p.y - start.y) > 3 * scale) {
				addLine(start, p);
				start = null;
			}
			return;
		}
		const line = hit(p);
		if (tool === 'select') selectLine(line?.id ?? null);
		if (tool === 'assign' && line && active) toggleLine(active.e, line.id);
	}

	/** Neue Linie - immer in die gewählte Ebene (bzw. Richtung) */
	function addLine(a: Point, b: Point) {
		const id = uniqueId('n', scene.lines);
		scene.lines.push({ id, point1: a, point2: b });
		if (target) {
			target.lineIds.push(id);
			expanded = new Set([...expanded, target.id]);
		}
		selectedId = id;
		changed();
	}

	async function selectLine(id: string | null) {
		selectedId = id;
		if (!id) return;
		const mine = membership.get(id) ?? [];
		if (mine.some((e) => !expanded.has(e))) expanded = new Set([...expanded, ...mine]);
		await tick();
		tree?.querySelector(`[data-line="${CSS.escape(id)}"]`)?.scrollIntoView({ block: 'nearest' });
	}

	function toggleLine(e: Ebene, lineId: string) {
		const i = e.lineIds.indexOf(lineId);
		if (i >= 0) e.lineIds.splice(i, 1);
		else e.lineIds.push(lineId);
		changed();
	}

	/** Linie aus allen Ebenen nehmen - sie bleibt als nicht zugeordnete Linie erhalten */
	function unassign(lineId: string) {
		for (const r of scene.richtungen) for (const e of r.ebenen) e.lineIds = e.lineIds.filter((x) => x !== lineId);
		changed();
	}

	function deleteLine(lineId: string) {
		unassign(lineId);
		scene.lines = scene.lines.filter((l) => l.id !== lineId);
		if (selectedId === lineId) selectedId = null;
	}

	/** Richtung wechseln: aus allen Ebenen nehmen, in die größte Ebene der Ziel-Richtung legen */
	function moveLine(lineId: string, richtungId: string) {
		const r = scene.richtungen.find((x) => x.id === richtungId);
		if (!r) return;
		unassign(lineId);
		if (!r.ebenen.length) r.ebenen.push({ id: nextEbeneId(r), name: 'ohne Ebene', lineIds: [] });
		r.ebenen.reduce((a, b) => (b.lineIds.length > a.lineIds.length ? b : a)).lineIds.push(lineId);
	}

	// ---------- Richtungen und Ebenen ----------
	function nextEbeneId(r: Richtung): string {
		const ids = new Set(scene.richtungen.flatMap((x) => x.ebenen.map((e) => e.id)));
		let n = r.ebenen.length + 1;
		while (ids.has(`${r.id}.${n}`)) n++;
		return `${r.id}.${n}`;
	}

	function activate(id: string | null) {
		[activeId, focus] = [id, null];
		if (id && !expanded.has(id)) expanded = new Set([...expanded, id]);
	}

	function toggleExpanded(id: string) {
		const next = new Set(expanded);
		if (!next.delete(id)) next.add(id);
		expanded = next;
	}

	function addRichtung() {
		const ids = scene.richtungen.flatMap((r) => [r, ...r.ebenen]);
		scene.richtungen.push({ id: uniqueId('g', ids), name: `Richtung ${scene.richtungen.length + 1}`, ebenen: [] });
		addEbene(scene.richtungen[scene.richtungen.length - 1]);
	}

	function addEbene(r: Richtung) {
		const id = nextEbeneId(r);
		r.ebenen.push({ id, name: `Ebene ${r.ebenen.length + 1}`, lineIds: [] });
		activate(id);
		if (tool === 'select') tool = 'draw';
		changed();
	}

	function removeFromLists(ids: Set<string>) {
		for (const L of LISTS) for (const g of scene[L.key]) g.ebenen = g.ebenen.filter((i) => !ids.has(i));
	}

	function deleteRichtung(r: Richtung) {
		const ids = new Set(r.ebenen.map((e) => e.id));
		scene.richtungen = scene.richtungen.filter((x) => x.id !== r.id);
		removeFromLists(ids);
		if (activeId && ids.has(activeId)) activeId = null;
		if (focus?.id === r.id) focus = null;
		changed();
	}

	function deleteEbene(r: Richtung, e: Ebene) {
		r.ebenen = r.ebenen.filter((x) => x.id !== e.id);
		removeFromLists(new Set([e.id]));
		if (activeId === e.id) activeId = null;
		changed();
	}

	/** Ebene in eine andere Richtung verschieben (ihre Linien gehen mit) */
	function moveEbene(ebeneId: string, to: Richtung) {
		const src = info.get(ebeneId);
		if (!src || src.r.id === to.id) return;
		const e = src.e;
		src.r.ebenen = src.r.ebenen.filter((x) => x.id !== ebeneId);
		to.ebenen.push(e);
		changed();
	}

	// ---------- Ebenen- und Senkrecht-Gruppen ----------
	function addGroup(key: ListKey, ebenen: string[] = []) {
		const L = LISTS.find((x) => x.key === key)!;
		const id = uniqueId(L.prefix, scene[key]);
		scene[key].push({ id, name: `${L.item} ${scene[key].length + 1}`, ebenen });
		focus = { kind: key, id };
		changed();
	}

	function deleteGroup(key: ListKey, g: EbenenGruppe) {
		scene[key] = scene[key].filter((x) => x.id !== g.id);
		if (focus?.id === g.id) focus = null;
		changed();
	}

	function addToGroup(g: EbenenGruppe, ids: string[]) {
		for (const id of ids) if (!g.ebenen.includes(id)) g.ebenen.push(id);
		changed();
	}

	function removeFromGroup(g: EbenenGruppe, id: string) {
		g.ebenen = g.ebenen.filter((x) => x !== id);
		changed();
	}

	// ---------- Drag & Drop ----------
	function dragStart(ev: DragEvent, kind: 'ebene' | 'richtung', id: string) {
		ev.dataTransfer?.setData('text/plain', `${kind}:${id}`);
		ev.stopPropagation();
	}

	function dropped(ev: DragEvent): { kind: string; id: string } | null {
		const [kind, ...rest] = (ev.dataTransfer?.getData('text/plain') ?? '').split(':');
		return rest.length ? { kind, id: rest.join(':') } : null;
	}

	/** Ebene -> sie selbst; Richtung -> Rule 5: alle ihre Ebenen, Rule 3/4: ihre größte Ebene */
	function ebenenOf(key: ListKey, item: { kind: string; id: string }): string[] {
		if (item.kind === 'ebene') return info.has(item.id) ? [item.id] : [];
		const r = scene.richtungen.find((x) => x.id === item.id);
		if (!r?.ebenen.length) return [];
		if (key === 'senkrecht') return r.ebenen.map((e) => e.id);
		return [r.ebenen.reduce((a, b) => (b.lineIds.length > a.lineIds.length ? b : a)).id];
	}

	function over(ev: DragEvent, id: string) {
		ev.preventDefault();
		dropTarget = id;
	}

	function drop(ev: DragEvent, action: (item: { kind: string; id: string }) => void) {
		ev.preventDefault();
		dropTarget = null;
		const item = dropped(ev);
		if (item) action(item);
	}

	function setTool(t: Tool) {
		[tool, start] = [t, null];
	}

	function onkeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
		if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) deleteLine(selectedId);
		if (e.key === 'Escape') {
			if (start) start = null;
			else [selectedId, activeId, focus] = [null, null, null];
		}
	}

	function stop(e: Event) {
		e.stopPropagation();
	}
</script>

<svelte:window {onkeydown} ondragend={() => (dropTarget = null)} />

<div class="view">
	<div class="canvas" class:scanning={busy === 'annotate' || busy === 'mllm'}>
		<Canvas
			{src}
			width={scene.imageWidth}
			height={scene.imageHeight}
			bind:scale
			dim
			cursor={tool === 'select' ? 'default' : 'crosshair'}
			{ondown}
			{onmove}
			{onup}
			{onclick}
			onhover={(p) => (cursor = p)}
		>
			{#each drawn as d (d.l.id)}
				{#if !d.hidden}
					{#if d.edge}
						<line class="edge" x1={d.l.point1.x} y1={d.l.point1.y} x2={d.l.point2.x} y2={d.l.point2.y} opacity={d.opacity} />
					{/if}
					<line
						x1={d.l.point1.x}
						y1={d.l.point1.y}
						x2={d.l.point2.x}
						y2={d.l.point2.y}
						stroke={d.color}
						stroke-width={d.width}
						stroke-dasharray={d.dash}
						opacity={d.opacity}
					/>
				{/if}
			{/each}
			{#if selected}
				<line class="selected" x1={selected.point1.x} y1={selected.point1.y} x2={selected.point2.x} y2={selected.point2.y} />
				{#each [selected.point1, selected.point2] as p}
					<rect class="handle" x={p.x - 5 * scale} y={p.y - 5 * scale} width={10 * scale} height={10 * scale} />
				{/each}
			{/if}
			{#if start}
				{#if cursor}<line class="draft" x1={start.x} y1={start.y} x2={cursor.x} y2={cursor.y} />{/if}
				<rect class="handle" x={start.x - 4 * scale} y={start.y - 4 * scale} width={8 * scale} height={8 * scale} />
			{/if}
		</Canvas>

		{#if busy === 'annotate' || busy === 'mllm'}
			<div class="scan">
				<span class="beam"></span>
				<p class="label">
					<b>···</b>
					{busy === 'annotate'
						? `Linien erkennen (${detectorName}) und gruppieren (TLC)`
						: `Linien erkennen (${detectorName})`}
				</p>
			</div>
		{:else if !scene.lines.length && !manual}
			<div class="empty-state panel">
				<strong>Wie soll annotiert werden?</strong>
				<label class="detector">
					<span class="mono">Linienerkennung</span>
					<select bind:value={detector}>
						{#each Object.entries(DETECTORS) as [id, d]}
							<option value={id}>{d.name} – {d.hint}</option>
						{/each}
					</select>
				</label>
				<div class="choices">
					<button class="btn" onclick={onannotate}>
						Automatisch annotieren lassen
					</button>
					<button class="btn" onclick={onmllm}>Mit MLLM annotieren</button>
					<button class="btn" onclick={() => ((manual = true), setTool('draw'))}>
						Manuell annotieren
					</button>
				</div>
			</div>
		{/if}
	</div>

	<aside>
		<div class="top">
			<section>
				<h3 class="label"> Werkzeug</h3>
				<div class="segments">
					{#each TOOLS as t}
						<button class:on={tool === t.id} onclick={() => setTool(t.id)}>{t.label}</button>
					{/each}
				</div>
				<p class="hint">
					{TOOLS.find((t) => t.id === tool)?.hint}
					{#if tool === 'draw'}
						{#if target}Ziel: <b>{label(target.id)}</b>{:else}<strong>Keine Ebene gewählt – die Linie bleibt frei.</strong>{/if}
					{:else if tool === 'assign' && !active}
						<strong>Zuerst eine Ebene aktivieren.</strong>
					{/if}
				</p>
			</section>

			<section class="card" class:empty={!selected}>
				{#if selected}
					{@const mine = membership.get(selected.id) ?? []}
					<h3 class="label"><b>·</b> Linie <span class="mono">{selected.id}</span> <span class="mono">{length(selected.id)} px</span></h3>
					<div class="chips">
						{#each mine as id}
							<span class="chip" style:border-color={dirColor(info.get(id)!.ri)}>
								{label(id)}<button title="aus der Ebene nehmen" onclick={() => toggleLine(info.get(id)!.e, selected.id)}>×</button>
							</span>
						{:else}
							<span class="hint">in keiner Ebene</span>
						{/each}
					</div>
					<div class="line-actions">
						<select
							value={info.get(mine[0] ?? '')?.r.id ?? ''}
							onchange={(e) => moveLine(selected.id, e.currentTarget.value)}
							title="Richtung wechseln"
						>
							<option value="" disabled>Richtung …</option>
							{#each scene.richtungen as r}<option value={r.id}>{r.name}</option>{/each}
						</select>
						{#if active && !mine.includes(active.e.id)}
							<button class="ghost" onclick={() => toggleLine(active.e, selected.id)}>+ aktive Ebene</button>
						{/if}
						<button class="ghost" disabled={!mine.length} onclick={() => unassign(selected.id)}>Lösen</button>
						<button class="ghost" onclick={() => deleteLine(selected.id)}>Löschen</button>
					</div>
				{:else}
					<h3 class="label"><b>·</b> Linie</h3>
					<p class="hint">Linie im Bild oder in der Liste einer Ebene anklicken.</p>
				{/if}
			</section>
		</div>

		<div class="cols">
			<section class="col" bind:this={tree}>
				<h3 class="label">
					 Parallele Linien nach Ebenen
					<button class="add" onclick={addRichtung}>+</button>
				</h3>
				<p class="hint">Gruppiere hier Linien, welche im Raum parallel verlaufen, und trenne sie nach flachen Ebenen wie die Seite eines Würfels. Kantenlinien können in mehreren Ebenen sein.</p>
				{#each scene.richtungen as r, ri (r.id)}
					{@const n = new Set(r.ebenen.flatMap((e) => e.lineIds)).size}
					<div
						class="richtung"
						class:drop={dropTarget === r.id}
						role="group"
						ondragover={(e) => over(e, r.id)}
						ondragleave={() => (dropTarget = null)}
						ondrop={(e) => drop(e, (it) => it.kind === 'ebene' && moveEbene(it.id, r))}
					>
						<div
							class="row r-row"
							class:on={focus?.kind === 'richtung' && focus.id === r.id}
							role="button"
							tabindex="0"
							draggable="true"
							ondragstart={(e) => dragStart(e, 'richtung', r.id)}
							onclick={() => ((focus = focus?.id === r.id ? null : { kind: 'richtung', id: r.id }), (activeId = null))}
							onkeydown={(e) => e.key === 'Enter' && (focus = { kind: 'richtung', id: r.id })}
						>
							<span class="swatch" style:background={dirColor(ri)}></span>
							{#if focus?.kind === 'richtung' && focus.id === r.id}
								<input type="text" bind:value={r.name} onclick={stop} oninput={changed} />
							{:else}
								<span class="name">{r.name}</span>
							{/if}
							<!-- <span class="mono count">{n}</span> -->
							<button class="mini" title="Ebene hinzufügen" onclick={(e) => (stop(e), addEbene(r))}>+</button>
							<button class="del" title="Richtung löschen" onclick={(e) => (stop(e), deleteRichtung(r))}>×</button>
						</div>
						{#each r.ebenen as e, ei (e.id)}
							<div
								class="row e-row"
								class:on={activeId === e.id}
								role="button"
								tabindex="0"
								draggable="true"
								ondragstart={(ev) => dragStart(ev, 'ebene', e.id)}
								onclick={() => activate(activeId === e.id ? null : e.id)}
								onkeydown={(ev) => ev.key === 'Enter' && activate(e.id)}
							>
								<button class="chev" title="Linien zeigen" onclick={(ev) => (stop(ev), toggleExpanded(e.id))}>
									{expanded.has(e.id) ? '▾' : '▸'}
								</button>
								<span class="plane" style:background={focusRichtung?.id === r.id ? PLANE_COLORS[ei % PLANE_COLORS.length] : 'transparent'}></span>
								{#if activeId === e.id}
									<input type="text" bind:value={e.name} onclick={stop} oninput={changed} />
								{:else}
									<span class="name">{e.name}</span>
								{/if}
								{#each badges(e.id) as b}<span class="badge mono">{b}</span>{/each}
								<!-- <span class="mono count">{e.lineIds.length}</span> -->
								<button class="del" title="Ebene löschen" onclick={(ev) => (stop(ev), deleteEbene(r, e))}>×</button>
							</div>
							{#if expanded.has(e.id)}
								<div class="lines">
									{#each e.lineIds as lid (lid)}
										<div
											class="row l-row"
											class:on={selectedId === lid}
											data-line={lid}
											role="button"
											tabindex="0"
											onclick={() => selectLine(selectedId === lid ? null : lid)}
											onkeydown={(ev) => ev.key === 'Enter' && selectLine(lid)}
										>
											<span class="mono">{lid}</span>
											<span class="mono len">{length(lid)} px</span>
											{#if (membership.get(lid)?.length ?? 0) > 1}<span class="badge mono">Kante</span>{/if}
											<button class="del" title="aus der Ebene nehmen" onclick={(ev) => (stop(ev), toggleLine(e, lid))}>×</button>
										</div>
									{:else}
										<p class="hint">Keine Linien – zeichnen oder zuordnen.</p>
									{/each}
								</div>
							{/if}
						{/each}
					</div>
				{:else}
					<p class="hint">Noch keine Richtung – „+ Richtung“ legt eine an (mit einer ersten Ebene).</p>
				{/each}
			</section>

			<section class="col">
				{#each LISTS as L}
					<div
						class="list"
						class:drop={dropTarget === `new-${L.key}`}
						role="group"
						ondragover={(e) => over(e, `new-${L.key}`)}
						ondragleave={() => (dropTarget = null)}
						ondrop={(e) => drop(e, (it) => addGroup(L.key, ebenenOf(L.key, it)))}
					>
						<h3 class="label">
							<b>{L.prefix.toUpperCase()}</b> {L.title} 
							<button class="add" onclick={() => addGroup(L.key)}>+</button>
						</h3>
						<p class="hint">{L.hint}</p>
						{#each scene[L.key] as g, k (g.id)}
							<div
								class="group"
								class:on={focus?.id === g.id}
								class:drop={dropTarget === g.id}
								role="button"
								tabindex="0"
								onclick={() => (focus = focus?.id === g.id ? null : { kind: L.key, id: g.id })}
								onkeydown={(e) => e.key === 'Enter' && (focus = { kind: L.key, id: g.id })}
								ondragover={(e) => (stop(e), over(e, g.id))}
								ondrop={(e) => (stop(e), drop(e, (it) => addToGroup(g, ebenenOf(L.key, it))))}
							>
								<div class="g-head">
									<span class="badge mono">{L.prefix.toUpperCase()}{k + 1}</span>
									{#if focus?.id === g.id}
										<input type="text" bind:value={g.name} onclick={stop} oninput={changed} />
									{:else}
										<span class="name">{g.name}</span>
									{/if}
									{#if active && !g.ebenen.includes(active.e.id)}
										<button class="mini" title="aktive Ebene hinzufügen" onclick={(e) => (stop(e), addToGroup(g, [active.e.id]))}>
											+ aktive
										</button>
									{/if}
									<button class="del" title="Gruppe löschen" onclick={(e) => (stop(e), deleteGroup(L.key, g))}>×</button>
								</div>
								<div class="chips">
									{#each g.ebenen as id}
										<span class="chip" style:border-color={info.has(id) ? dirColor(info.get(id)!.ri) : POOL}>
											{label(id)}<button title="aus der Gruppe nehmen" onclick={(e) => (stop(e), removeFromGroup(g, id))}>×</button>
										</span>
									{:else}
										<span class="hint">Ebenen hierher ziehen</span>
									{/each}
								</div>
							</div>
						{/each}
					</div>
				{/each}
			</section>
		</div>

		<footer>
			<!-- <div class="opts">
				<label class="check"><input type="checkbox" bind:checked={onlyFocus} /> nur Auswahl zeigen</label>
				<span class="mono stats">{scene.lines.length} Linien </span>
			</div> -->
			<button class="ghost" disabled={!!busy} onclick={onannotate} title="{detectorName} + TLC v2">Automatisch annotieren</button>
			<button class="ghost" disabled={!!busy} onclick={onmllm} title="{detectorName} + Gruppierung durch ein MLLM">
				Mit MLLM annotieren
			</button>
			<button class="btn" disabled={!!busy || !scene.richtungen.length} onclick={onevaluate}>
				{busy === 'evaluate' ? 'Wird ausgewertet …' : 'Auswerten'}
			</button>
		</footer>
	</aside>
</div>

<style>
	.view {
		display: grid;
		grid-template-columns: minmax(0, 1fr) min(720px, 54vw);
		height: 100%;
		min-height: 0;
	}
	.canvas {
		position: relative;
		min-height: 0;
	}
	line {
		vector-effect: non-scaling-stroke;
		pointer-events: none;
	}
	.edge {
		stroke: white;
		stroke-width: 6;
	}
	.selected {
		stroke: white;
		stroke-width: 5;
		opacity: 0.6;
	}
	.handle {
		fill: white;
		stroke: var(--ink);
		stroke-width: 1.5;
		vector-effect: non-scaling-stroke;
	}
	.draft {
		stroke: white;
		stroke-width: 2;
		stroke-dasharray: 6 4;
	}

	/* Scan-Animation während der Linienerkennung */
	.canvas.scanning :global(svg) {
		filter: grayscale(1) contrast(1.1);
	}
	.scan {
		position: absolute;
		inset: 0;
		overflow: hidden;
		pointer-events: none;
	}
	.beam {
		position: absolute;
		left: 0;
		right: 0;
		top: 0;
		height: 2px;
		background: var(--accent);
		box-shadow: 0 0 16px 4px color-mix(in srgb, var(--accent) 50%, transparent);
		animation: scan 1.8s cubic-bezier(0.65, 0, 0.35, 1) infinite alternate;
	}
	@keyframes scan {
		to {
			top: calc(100% - 2px);
		}
	}
	.scan .label {
		position: absolute;
		left: 50%;
		bottom: 1.5rem;
		transform: translateX(-50%);
		margin: 0;
		padding: 0.7rem 1rem;
		background: rgba(20, 23, 28, 0.85);
		color: var(--paper);
	}
	.scan .label::after {
		display: none;
	}

	.empty-state {
		position: absolute;
		left: 50%;
		top: 50%;
		transform: translate(-50%, -50%);
		width: min(380px, 90%);
		padding: 1.5rem;
		text-align: center;
	}
	.empty-state strong {
		font-size: 18px;
		letter-spacing: 0.03em;
	}
	.detector {
		display: grid;
		gap: 0.35rem;
		margin-top: 1.2rem;
		text-align: left;
	}
	.detector span {
		font-size: 11px;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.detector select {
		width: 100%;
		padding: 0.55em 0.7em;
		border: 1px solid var(--ink);
		border-radius: 0;
		background: var(--panel);
		color: var(--ink);
		font: inherit;
		font-size: 13px;
		cursor: pointer;
	}
	.detector select:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
	}
	.choices {
		display: grid;
		gap: 0.6rem;
		margin-top: 1rem;
	}
	.choices small {
		display: block;
		margin-top: 0.25rem;
		font: 400 11px var(--mono);
		letter-spacing: 0.02em;
		text-transform: none;
		opacity: 0.6;
	}

	/* Seitenleiste: oben Werkzeug + Linie, zwei Spalten, unten Aktionen - jeweils über die volle Breite */
	aside {
		display: grid;
		grid-template-rows: auto minmax(0, 1fr) auto;
		min-height: 0;
		min-width: 0;
		border-left: 1px solid var(--hair);
		background: var(--panel);
	}
	.top {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 1.25rem;
		padding: 1rem 1.25rem;
		border-bottom: 1px solid var(--ink);
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		min-height: 0;
	}
	.col {
		overflow-y: auto;
		padding: 0.9rem 1rem 1rem 1.25rem;
	}
	.col + .col {
		border-left: 1px solid var(--hair);
		padding-left: 1rem;
	}
	.label .add {
		margin-left: auto;
		order: 3;
	}
	.hint {
		margin-top: 0.5rem;
		margin-bottom: 0.5rem;
		font-size: 14px;
		color: var(--muted);
	}
	.hint strong {
		color: var(--accent);
		font-weight: 600;
	}
	.hint b {
		color: var(--ink);
		font-weight: 600;
	}
	.add,
	.mini {
		padding: 0.15em 0.55em;
		border: 1px solid var(--ink);
		background: transparent;
		font-size: 12px;
		letter-spacing: 0;
		text-transform: none;
		color: var(--ink);
	}
	.add:hover,
	.mini:hover {
		background: var(--accent);
		border-color: var(--accent);
		color: white;
	}

	.richtung {
		margin-bottom: 0.6rem;
		border: 1px solid transparent;
	}
	.drop {
		border-color: var(--accent) !important;
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.3rem 0.4rem;
		cursor: pointer;
		font-size: 14px;
	}
	.row:hover {
		background: rgba(20, 23, 28, 0.04);
	}
	.r-row {
		border-bottom: 1px solid var(--ink);
		font-weight: 600;
	}
	.e-row {
		padding-left: 0.3rem;
		border-bottom: 1px solid var(--hair);
	}
	.l-row {
		padding: 0.15rem 0.4rem 0.15rem 2.4rem;
		font-size: 12px;
	}
	.l-row .len {
		flex: 1;
		color: var(--muted);
	}
	.row.on {
		background: var(--ink);
		color: var(--paper);
	}
	.row.on .len {
		color: inherit;
		opacity: 0.6;
	}
	.row.on input {
		border-color: var(--paper);
		color: var(--paper);
	}
	.lines {
		border-bottom: 1px solid var(--hair);
	}
	.lines .hint {
		margin: 0.3rem 0 0.4rem 2.4rem;
		font-size: 12px;
	}
	.chev {
		width: 1.1rem;
		border: 0;
		background: transparent;
		padding: 0;
		font-size: 11px;
		color: inherit;
	}
	.swatch {
		width: 11px;
		height: 11px;
		flex: none;
		transform: rotate(45deg);
	}
	.plane {
		width: 8px;
		height: 8px;
		flex: none;
		border: 1px solid currentColor;
	}
	.name,
	.row input,
	.g-head input,
	.g-head .name {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.count {
		font-size: 12px;
		opacity: 0.6;
	}
	.badge {
		padding: 0 0.3em;
		border: 1px solid currentColor;
		font-size: 10px;
		opacity: 0.75;
	}
	.del {
		border: 0;
		background: transparent;
		padding: 0 0.2rem;
		opacity: 0.45;
	}
	.del:hover {
		opacity: 1;
		color: var(--bad);
	}
	.row.on .mini {
		border-color: var(--paper);
		color: var(--paper);
	}

	.list + .list {
		margin-top: 1.2rem;
	}
	.list {
		border: 1px solid transparent;
	}
	.group {
		margin-bottom: 0.5rem;
		padding: 0.4rem 0.5rem;
		border: 1px solid var(--hair);
		cursor: pointer;
	}
	.group.on {
		border-color: var(--ink);
		box-shadow: inset 4px 0 0 var(--ink);
	}
	.g-head {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 14px;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		margin-top: 0.4rem;
	}
	.chip {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		max-width: 100%;
		min-width: 0;
		overflow-wrap: anywhere;
		padding: 0.05em 0.2em 0.05em 0.45em;
		border: 1px solid;
		border-left-width: 4px;
		font-size: 12px;
		background: var(--panel);
	}
	.chip button {
		border: 0;
		background: transparent;
		padding: 0 0.2em;
		opacity: 0.5;
	}
	.chip button:hover {
		opacity: 1;
		color: var(--bad);
	}
	.card {
		padding: 0.6rem 0.7rem;
		border: 1px solid var(--ink);
		min-width: 0;
	}
	.card.empty {
		border-style: dashed;
		border-color: var(--hair);
	}
	.card .label {
		margin-bottom: 0.4rem;
	}
	.line-actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		margin-top: 0.6rem;
	}
	.line-actions .ghost {
		padding: 0.3em 0.6em;
		font-size: 12px;
	}
	select {
		font: inherit;
		font-size: 12px;
		border: 1px solid var(--ink);
		background: transparent;
		padding: 0.25em 0.3em;
		border-radius: 0;
	}

	footer {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		align-items: center;
		gap: 0.6rem 0.8rem;
		padding: 0.8rem 1.25rem 1rem;
		border-top: 1px solid var(--ink);
	}
	.opts {
		grid-column: 1 / -1;
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	.check {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 13px;
		white-space: nowrap;
	}
	.stats {
		font-size: 12px;
		color: var(--muted);
	}
	footer .ghost,
	footer .btn {
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
</style>
