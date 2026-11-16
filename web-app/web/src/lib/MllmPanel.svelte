<script lang="ts">
	import { base } from '$app/paths';
	import { env } from '$env/dynamic/public';
	import { mllmStatus, type MllmStatus } from './api';
	import type { Scene } from './types';

	interface Props {
		token: string;
		/** Anzahl erkannter / nummerierter Linien */
		lines: number;
		shown: number;
		/** Name der Linienerkennung */
		detector: string;
		onaccept: (scene: Scene) => void;
		onclose: () => void;
	}

	let { token, lines, shown, detector, onaccept, onclose }: Props = $props();

	const url = $derived(`${(env.PUBLIC_URL || location.origin + base).replace(/\/$/, '')}/group/${token}`);
	const prompt = $derived(
		`Öffne ${url} und lies die Anleitung. Sieh dir die Bilder dort genau an (auch die vergrößerten ` +
		`Ausschnitte), gruppiere die nummerierten Linien wie beschrieben und gib die Gruppierung per POST als JSON ` +
		`an dieselbe Adresse ab. Prüfe die Hinweise in der Antwort und gib bei Bedarf erneut ab.`
	);
	const local = $derived(/^(localhost|127\.|\[::1\])/.test(new URL(url).hostname));

	let status = $state<MllmStatus>({ status: 'offen', version: 0 });
	let error = $state('');
	let copied = $state('');

	$effect(() => {
		const poll = async () => {
			try {
				status = await mllmStatus(token);
				error = '';
			} catch (e) {
				error = (e as Error).message;
			}
		};
		poll();
		const id = setInterval(poll, 2000);
		return () => clearInterval(id);
	});

	async function copy(text: string, what: string) {
		await navigator.clipboard.writeText(text);
		copied = what;
		setTimeout(() => (copied = ''), 1500);
	}

	const counts = $derived.by(() => {
		const s = status.scene;
		return {
			richtungen: s?.richtungen.length ?? 0,
			ebenen: s?.richtungen.reduce((n, r) => n + r.ebenen.length, 0) ?? 0,
			horizonte: s?.horizonte.length ?? 0,
			senkrecht: s?.senkrecht.length ?? 0
		};
	});
</script>

<div class="backdrop">
	<div class="dialog panel">
		<h3 class="label"><b>MLLM</b> Gruppierung durch ein Sprachmodell</h3>
		<p class="intro">
			{detector} hat {lines} Linien erkannt, die {shown} längsten sind nummeriert. Gib die Adresse einem MLLM mit
			Web-Zugriff – es sieht sich dort Bild und Linien an und gibt die Gruppierung per POST ab.
		</p>

		<div class="field">
			<span class="mono">Adresse</span>
			<code>{url}</code>
			<button class="ghost" onclick={() => copy(url, 'url')}>{copied === 'url' ? 'Kopiert' : 'Kopieren'}</button>
			<a class="ghost" href={url} target="_blank" rel="noopener">Öffnen</a>
		</div>
		<div class="field prompt">
			<span class="mono">Prompt</span>
			<p>{prompt}</p>
			<button class="ghost" onclick={() => copy(prompt, 'prompt')}>{copied === 'prompt' ? 'Kopiert' : 'Kopieren'}</button>
		</div>
		{#if local}
			<p class="warn">
				Die Adresse ist nur auf diesem Rechner erreichbar. Für ein MLLM in der Cloud den Webserver öffentlich
				machen (z.B. Tunnel) und PUBLIC_URL in .env setzen.
			</p>
		{/if}

		<div class="status" class:done={status.status === 'abgegeben'}>
			{#if status.status === 'offen'}
				<span class="spinner"></span> Warte auf die Abgabe …
			{:else}
				<div>
					<strong>Abgabe {status.version}</strong> um {status.abgegeben}{status.modell ? ` · ${status.modell}` : ''}
					<div class="mono counts">
						{counts.richtungen} Richtungen · {counts.ebenen} Ebenen · {counts.horizonte} Ebenen-Gruppen ·
						{counts.senkrecht} Senkrecht-Gruppen
					</div>
					{#if status.notiz}<p class="note">„{status.notiz}“</p>{/if}
					{#if status.hinweise?.length}
						<ul>
							{#each status.hinweise as h}<li>{h}</li>{/each}
						</ul>
					{/if}
				</div>
			{/if}
		</div>
		{#if error}<p class="warn">{error}</p>{/if}

		<div class="actions">
			<button class="ghost" onclick={onclose}>Schließen</button>
			<button class="btn" disabled={!status.scene} onclick={() => status.scene && onaccept(status.scene)}>
				In den Editor übernehmen
			</button>
		</div>
	</div>
</div>

<style>
	.backdrop {
		position: fixed;
		inset: 56px 0 0 0;
		display: grid;
		place-items: center;
		padding: 1rem;
		background: rgba(20, 23, 28, 0.55);
		z-index: 10;
	}
	.dialog {
		width: min(680px, 100%);
		max-height: 100%;
		overflow-y: auto;
		padding: 1.5rem;
	}
	.intro {
		margin: 0 0 1rem;
		font-size: 14px;
		color: var(--muted);
	}
	.field {
		display: grid;
		grid-template-columns: 4.5rem 1fr auto auto;
		align-items: center;
		gap: 0.6rem;
		padding: 0.6rem 0;
		border-top: 1px solid var(--hair);
	}
	.field > span {
		font-size: 11px;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.field code {
		font: 13px var(--mono);
		overflow-wrap: anywhere;
	}
	.field a {
		color: inherit;
		text-decoration: none;
	}
	.prompt {
		grid-template-columns: 4.5rem 1fr auto;
		border-bottom: 1px solid var(--hair);
	}
	.prompt p {
		margin: 0;
		font-size: 13px;
	}
	.warn {
		margin: 0.8rem 0 0;
		padding: 0.5rem 0.8rem;
		border-left: 4px solid var(--warn);
		background: rgba(201, 154, 0, 0.1);
		font-size: 13px;
	}
	.status {
		display: flex;
		align-items: center;
		gap: 0.8rem;
		margin-top: 1rem;
		padding: 0.9rem 1rem;
		border: 1px solid var(--ink);
		font-size: 14px;
	}
	.status.done {
		border-left: 5px solid var(--ok);
	}
	.spinner {
		width: 12px;
		height: 12px;
		background: var(--accent);
		animation: spin 1.2s cubic-bezier(0.65, 0, 0.35, 1) infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(180deg);
		}
	}
	.counts {
		margin-top: 0.2rem;
		font-size: 12px;
		color: var(--muted);
	}
	.note {
		margin: 0.4rem 0 0;
		font-size: 13px;
		font-style: italic;
	}
	.status ul {
		margin: 0.5rem 0 0;
		padding-left: 1.1rem;
		font-size: 13px;
		color: var(--muted);
	}
	.actions {
		display: flex;
		justify-content: flex-end;
		gap: 0.6rem;
		margin-top: 1.2rem;
	}
</style>
