<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { Point } from './types';

	interface Props {
		src: string;
		width: number;
		height: number;
		/** true = Ziehen übernommen (z.B. Linie zeichnen), sonst wird verschoben */
		ondown?: (p: Point, e: PointerEvent) => boolean;
		onmove?: (p: Point) => void;
		onup?: (p: Point) => void;
		/** Klick ohne Verschieben */
		onclick?: (p: Point) => void;
		/** Mausbewegung ohne gedrückte Taste (z.B. Vorschau beim Zeichnen) */
		onhover?: (p: Point) => void;
		cursor?: string;
		/** Bild abdunkeln, damit Linien besser sichtbar sind */
		dim?: boolean;
		/** Bild-Einheiten pro Bildschirm-Pixel */
		scale?: number;
		children?: Snippet;
	}

	let {
		src,
		width,
		height,
		ondown,
		onmove,
		onup,
		onclick,
		onhover,
		cursor = 'grab',
		dim = false,
		scale = $bindable(1),
		children
	}: Props = $props();

	let svg: SVGSVGElement;
	let w = $state(0);
	let h = $state(0);
	let center = $state({ x: 0, y: 0 });

	const viewBox = $derived(`${center.x - (scale * w) / 2} ${center.y - (scale * h) / 2} ${scale * w} ${scale * h}`);

	/** Ausschnitt so setzen, dass der Bereich ganz sichtbar ist */
	export function fit(box = { x: 0, y: 0, w: width, h: height }) {
		center = { x: box.x + box.w / 2, y: box.y + box.h / 2 };
		scale = Math.max(box.w / w, box.h / h) * 1.06 || 1;
	}

	let fitted = false;
	$effect(() => {
		if (w > 0 && h > 0 && !fitted) {
			fitted = true;
			fit();
		}
	});

	function toImage(e: { clientX: number; clientY: number }): Point {
		const r = svg.getBoundingClientRect();
		return {
			x: center.x + (e.clientX - r.left - w / 2) * scale,
			y: center.y + (e.clientY - r.top - h / 2) * scale
		};
	}

	let pan = $state<{ x: number; y: number; center: Point } | null>(null);
	let taken = false;
	let moved = false;
	let start = { x: 0, y: 0 };

	function down(e: PointerEvent) {
		svg.setPointerCapture(e.pointerId);
		start = { x: e.clientX, y: e.clientY };
		moved = false;
		taken = e.button === 0 && !!ondown?.(toImage(e), e);
		if (!taken) pan = { x: e.clientX, y: e.clientY, center: { ...center } };
	}

	function move(e: PointerEvent) {
		if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 4) moved = true;
		if (pan) {
			center = { x: pan.center.x - (e.clientX - pan.x) * scale, y: pan.center.y - (e.clientY - pan.y) * scale };
		} else if (taken) {
			onmove?.(toImage(e));
		} else {
			onhover?.(toImage(e));
		}
	}

	function up(e: PointerEvent) {
		if (taken) onup?.(toImage(e));
		else if (!moved && e.button === 0) onclick?.(toImage(e));
		pan = null;
		taken = false;
	}

	function wheel(e: WheelEvent) {
		e.preventDefault();
		const p = toImage(e);
		const f = Math.exp(e.deltaY * 0.0015);
		center = { x: p.x - (p.x - center.x) * f, y: p.y - (p.y - center.y) * f };
		scale *= f;
	}
</script>

<div class="stage" bind:clientWidth={w} bind:clientHeight={h}>
	<svg
		bind:this={svg}
		{viewBox}
		style:cursor={pan ? 'grabbing' : cursor}
		role="application"
		aria-label="Bild - Mausrad zoomt, Ziehen verschiebt"
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onpointercancel={() => ((pan = null), (taken = false))}
		onwheel={wheel}
		oncontextmenu={(e) => e.preventDefault()}
	>
		<image href={src} {width} {height} opacity={dim ? 0.55 : 1} />
		<rect class="frame" {width} {height} />
		{@render children?.()}
	</svg>
	<button class="reset" title="Ganzes Bild zeigen" onclick={() => fit()}>⤢</button>
</div>

<style>
	.stage {
		position: relative;
		width: 100%;
		height: 100%;
		overflow: hidden;
		background-color: var(--stage);
		background-image:
			linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
			linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px);
		background-size: 24px 24px;
	}
	svg {
		display: block;
		width: 100%;
		height: 100%;
		touch-action: none;
		user-select: none;
	}
	.frame {
		fill: none;
		stroke: rgba(255, 255, 255, 0.35);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.reset {
		position: absolute;
		right: 12px;
		bottom: 12px;
		width: 34px;
		height: 34px;
		border: 1px solid rgba(255, 255, 255, 0.5);
		background: rgba(20, 23, 28, 0.7);
		color: white;
		font-size: 16px;
	}
	.reset:hover {
		background: var(--accent);
		border-color: var(--accent);
	}
</style>
