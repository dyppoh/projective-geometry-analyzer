<script lang="ts">
	import { asset } from '$app/paths';
	import { annotate, evaluate, imageSize, readAsDataUrl, startMllm } from '$lib/api';
	import Befunde from '$lib/Befunde.svelte';
	import Editor from '$lib/Editor.svelte';
	import InfoPanels from '$lib/InfoPanels.svelte';
	import MllmPanel from '$lib/MllmPanel.svelte';
	import { exportJson, importJson } from '$lib/transfer';
	import type { Detector, Result, Scene } from '$lib/types';

	const TESTBILDER = Object.entries(
		import.meta.glob<string>('$lib/testbilder/*.{jpg,jpeg,png,webp}', { eager: true, query: '?url', import: 'default' })
	).map(([path, url]) => ({ name: path.split('/').pop()!.replace(/\.\w+$/, ''), url }));

	let image = $state<{ src: string; name: string } | null>(null);
	let result = $state<Result | null>(null);
	/** bearbeitbare Linien und Gruppen */
	let scene = $state<Scene | null>(null);
	/** seit der letzten Auswertung geändert */
	let dirty = $state(false);
	let busy = $state<'' | 'annotate' | 'mllm' | 'evaluate'>('');
	/** Linienerkennung für automatisch / MLLM */
	let detector = $state<Detector>('mlsd');
	/** offene MLLM-Aufgabe */
	let mllm = $state<{ token: string; lines: number; shown: number; detector: string } | null>(null);
	let error = $state('');
	let tab = $state<'editor' | 'befunde'>('editor');
	let dragOver = $state(false);
	/** Hinweis „nur am Desktop“ auf kleinen Bildschirmen weggeklickt */
	let mobileOk = $state(false);

	async function start(blob: Blob, name: string) {
		error = '';
		try {
			const src = await readAsDataUrl(blob);
			const { width, height } = await imageSize(src);
			open(src, name, { imageWidth: width, imageHeight: height, lines: [], richtungen: [], horizonte: [], senkrecht: [] });
		} catch (e) {
			error = `Bild nicht lesbar: ${(e as Error).message}`;
		}
	}

	function open(src: string, name: string, s: Scene) {
		image = { src, name };
		scene = s;
		[result, dirty, tab, mllm] = [null, false, 'editor', null];
	}

	async function load(file: File) {
		error = '';
		try {
			const { name, src, scene } = await importJson(file, file.name.replace(/\.json$/i, ''));
			open(src, name, scene);
		} catch (e) {
			error = `JSON nicht lesbar: ${(e as Error).message}`;
		}
	}

	async function autoAnnotate() {
		if (!image || !scene) return;
		if (scene.lines.length && !confirm('Vorhandene Linien und Gruppen ersetzen?')) return;
		busy = 'annotate';
		try {
			scene = await annotate(image.src, detector);
			dirty = true;
		} catch (e) {
			alert((e as Error).message);
		} finally {
			busy = '';
		}
	}

	async function groupWithMllm() {
		if (!image) return;
		busy = 'mllm';
		try {
			mllm = await startMllm(image.src, detector);
		} catch (e) {
			alert((e as Error).message);
		} finally {
			busy = '';
		}
	}

	function acceptMllm(s: Scene) {
		if (scene?.lines.length && !confirm('Vorhandene Linien und Gruppen ersetzen?')) return;
		[scene, dirty, mllm] = [s, true, null];
	}

	async function run() {
		if (!scene) return;
		busy = 'evaluate';
		try {
			result = await evaluate($state.snapshot(scene));
			[dirty, tab] = [false, 'befunde'];
		} catch (e) {
			alert((e as Error).message);
		} finally {
			busy = '';
		}
	}

	function reset() {
		if (scene?.lines.length && !confirm('Bild schließen und Linien verwerfen?')) return;
		[image, result, scene, dirty, mllm] = [null, null, null, false, null];
	}

	function onfile(files: FileList | null | undefined) {
		const f = files?.[0];
		if (!f) return;
		if (f.type === 'application/json' || /\.json$/i.test(f.name)) load(f);
		else if (f.type.startsWith('image/')) start(f, f.name.replace(/\.\w+$/, ''));
		else error = 'Bitte ein Bild oder eine exportierte JSON-Datei wählen.';
	}

	async function pickTest(t: { url: string; name: string }) {
		start(await (await fetch(t.url)).blob(), t.name);
	}
</script>

<div class="app">
	<header>
		<button class="brand" onclick={reset} title="Startseite" >
			<img src={asset('/favicon.svg')} width="22" height="22" alt="Favicon" />
			<!-- <span>Perspektivische Analyse</span> -->
		</button>

		{#if scene && image}
			<nav>
				<button class:on={tab === 'editor'} onclick={() => (tab = 'editor')}>
					<b>01</b> Linien &amp; Gruppen{#if dirty && result}<i title="seit der Auswertung geändert"></i>{/if}
				</button>
				<button class:on={tab === 'befunde'} disabled={!result} onclick={() => (tab = 'befunde')}>
					<b>02</b> Befunde
				</button>
			</nav>
			<span class="file mono">{image.name}</span>
			<button
				class="ghost"
				onclick={() => exportJson(image!.name, image!.src, $state.snapshot(scene!))}
				title="Bild, Linien und Gruppen als JSON speichern"
			>
				JSON exportieren
			</button>
		{/if}
	</header>

	<main>
		{#if scene && image}
			{#if tab === 'befunde' && result}
				<Befunde src={image.src} {result} stale={dirty} />
			{:else}
				<Editor
					src={image.src}
					bind:scene
					bind:dirty
					bind:detector
					{busy}
					onannotate={autoAnnotate}
					onmllm={groupWithMllm}
					onevaluate={run}
				/>
			{/if}
			{#if mllm}
				<MllmPanel {...mllm} onaccept={acceptMllm} onclose={() => (mllm = null)} />
			{/if}
			{#if !mobileOk}
				<div class="mobile-block" role="dialog" aria-modal="true" aria-labelledby="mobile-block-title">
					<div class="panel">
						<span class="tag mono">Nur am Desktop</span>
						<h2 id="mobile-block-title">Der Editor braucht einen größeren Bildschirm</h2>
						<p>
							Linien zeichnen, Gruppen zuordnen und die Befunde ansehen funktioniert auf dem Handy nicht
							sinnvoll. Öffne die Seite am besten am Laptop oder Desktop.
						</p>
						<div class="actions">
							<button class="btn" onclick={reset}>Zur Startseite</button>
							<button class="ghost" onclick={() => (mobileOk = true)}>Trotzdem anzeigen</button>
						</div>
					</div>
				</div>
			{/if}
		{:else}
			<div class="home">
			<div class="start">
				<p class="hint">
					Der Editor ist für große Bildschirme gebaut – zum Annotieren und Auswerten die Seite am besten am
					Desktop öffnen.
				</p>

				<section>
					<label
						class="drop panel"
						class:over={dragOver}
						ondragover={(e) => (e.preventDefault(), (dragOver = true))}
						ondragleave={() => (dragOver = false)}
						ondrop={(e) => (e.preventDefault(), (dragOver = false), onfile(e.dataTransfer?.files))}
					>
						<input
							type="file"
							accept="image/*,.json,application/json"
							onchange={(e) => onfile(e.currentTarget.files)}
							hidden
						/>
						<strong class="pointer">Bild oder JSON hierher ziehen</strong>
						<span class="pointer">oder klicken zum Auswählen</span>
						<strong class="touch">Bild oder JSON auswählen</strong>
						<span class="touch">aus der Galerie oder per Kamera</span>
					</label>
				</section>

				<section>
					<h2 class="label">Testbilder</h2>
					<div class="tests">
						{#each TESTBILDER as t}
							<button class="test" onclick={() => pickTest(t)}>
								<img src={t.url} alt={t.name} loading="lazy" />
								<span class="mono">{t.name}</span>
							</button>
						{/each}
					</div>
				</section>

				{#if error}
					<p class="error">{error}</p>
				{/if}

				<div class="infos">
					<InfoPanels />
				</div>
			</div>
			</div>
		{/if}
	</main>
</div>

<style>
	.app {
		display: grid;
		grid-template-rows: auto 1fr;
		height: 100vh;
		height: 100dvh;
	}
	header {
		display: flex;
		align-items: center;
		gap: 1.5rem;
		height: 56px;
		padding: 0 1.25rem;
		background: var(--ink);
		color: var(--paper);
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		border: 0;
		background: transparent;
		padding: 0;
		font-size: 17px;
		font-weight: 700;
		letter-spacing: 0.22em;
		text-transform: uppercase;
	}
	.tagline {
		color: rgba(242, 241, 236, 0.55);
		font-size: 13px;
	}
	nav {
		display: flex;
		align-self: stretch;
		margin-left: 1rem;
	}
	nav button {
		position: relative;
		display: flex;
		align-items: center;
		gap: 0.6rem;
		padding: 0 1.3rem;
		border: 0;
		border-bottom: 3px solid transparent;
		background: transparent;
		color: rgba(242, 241, 236, 0.6);
		font-size: 14px;
		letter-spacing: 0.04em;
	}
	nav button:not(:disabled):hover {
		color: var(--paper);
	}
	nav button.on {
		color: var(--paper);
		border-bottom-color: var(--accent);
		background: rgba(255, 255, 255, 0.06);
	}
	nav b {
		font: 600 11px var(--mono);
		color: var(--accent);
	}
	nav i {
		width: 7px;
		height: 7px;
		background: var(--warn);
		transform: rotate(45deg);
	}
	.file {
		margin-left: auto;
		font-size: 12px;
		color: rgba(242, 241, 236, 0.55);
	}
	header .ghost {
		border-color: var(--paper);
	}
	header .ghost:hover {
		background: var(--paper);
		color: var(--ink);
	}
	main {
		min-height: 0;
	}

	/* Start */
	.home {
		height: 100%;
		overflow-y: auto;
	}
	.infos {
		grid-column: 1 / -1;
		display: grid;
		gap: 2.5rem;
		padding-top: 1rem;
	}
	.start {
		display: grid;
		grid-template-columns: minmax(280px, 1fr) minmax(0, 1.4fr);
		gap: 3rem;
		max-width: 1200px;
		margin: 0 auto;
		padding: 3.5rem 2rem;
	}
	.drop {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 0.4rem;
		aspect-ratio: 1;
		cursor: pointer;
		transition: border-color 0.15s;
	}
	.drop svg {
		margin-bottom: 1rem;
	}
	.drop strong {
		font-size: 18px;
		letter-spacing: 0.03em;
	}
	.drop span {
		color: var(--muted);
		font-size: 14px;
	}
	.drop .sub {
		margin-top: 0.3rem;
		font-size: 12px;
	}
	.drop:hover,
	.drop.over {
		border-color: var(--accent);
	}
	.drop.over {
		background: color-mix(in srgb, var(--accent) 6%, transparent);
	}
	.tests {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 1rem;
	}
	.test {
		display: flex;
		flex-direction: column;
		padding: 0;
		border: 1px solid var(--hair);
		background: var(--panel);
		text-align: left;
	}
	.test img {
		display: block;
		width: 100%;
		aspect-ratio: 1;
		object-fit: cover;
		filter: grayscale(0.35);
		transition: filter 0.15s;
	}
	.test span {
		padding: 0.5rem 0.6rem;
		border-top: 1px solid var(--hair);
		font-size: 12px;
	}
	.test:hover {
		border-color: var(--ink);
	}
	.test:hover img {
		filter: none;
	}
	.test:hover span {
		background: var(--ink);
		color: var(--paper);
	}
	.error {
		grid-column: 1 / -1;
		margin: 0;
		padding: 0.8rem 1rem;
		border-left: 4px solid var(--bad);
		background: rgba(224, 53, 43, 0.08);
	}

	/* Startseite auf Touch-Geräten und schmalen Bildschirmen (der Editor bleibt Desktop-only) */
	.touch,
	.hint,
	.mobile-block {
		display: none;
	}
	@media (hover: none) {
		.pointer {
			display: none;
		}
		.touch {
			display: inline;
		}
	}
	@media (max-width: 760px) {
		header {
			gap: 1rem;
			padding: 0 1rem;
		}
		.brand {
			gap: 0.55rem;
			font-size: 13px;
			letter-spacing: 0.14em;
		}
		.brand img {
			width: 26px;
			height: 26px;
		}
		.start {
			grid-template-columns: minmax(0, 1fr);
			gap: 2rem;
			padding: 1.25rem 1rem 2.5rem;
		}
		.hint {
			display: block;
			margin: 0;
			padding: 0.7rem 0.9rem;
			border-left: 3px solid var(--accent);
			background: color-mix(in srgb, var(--accent) 7%, transparent);
			font-size: 13px;
			line-height: 1.45;
		}
		.drop {
			aspect-ratio: auto;
			min-height: 150px;
			padding: 1.5rem 1rem;
			text-align: center;
		}
		.drop strong {
			font-size: 16px;
		}
		.tests {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 0.75rem;
		}
		.test span {
			overflow: hidden;
			text-overflow: ellipsis;
			white-space: nowrap;
			font-size: 11px;
		}
		.infos {
			gap: 2rem;
			padding-top: 0.5rem;
		}

		/* Meldung statt Editor */
		.mobile-block {
			position: fixed;
			inset: 0;
			z-index: 100;
			display: flex;
			align-items: center;
			justify-content: center;
			padding: 1rem;
			background: color-mix(in srgb, var(--ink) 55%, transparent);
			backdrop-filter: blur(3px);
		}
		.mobile-block .panel {
			width: 100%;
			max-width: 420px;
			padding: 1.4rem 1.25rem 1.25rem;
		}
		.mobile-block .tag {
			display: inline-block;
			margin-bottom: 0.8rem;
			padding: 0.2em 0.5em;
			background: var(--accent);
			color: white;
			font-size: 11px;
			letter-spacing: 0.08em;
			text-transform: uppercase;
		}
		.mobile-block h2 {
			font-size: 18px;
			line-height: 1.3;
		}
		.mobile-block p {
			margin: 0.6rem 0 1.2rem;
			color: var(--muted);
			font-size: 14px;
			line-height: 1.5;
		}
		.mobile-block .actions {
			display: flex;
			flex-wrap: wrap;
			align-items: center;
			gap: 0.75rem;
		}
	}
</style>
