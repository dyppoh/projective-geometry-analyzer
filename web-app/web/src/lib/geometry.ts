import type { Point } from './types';

export const PALETTE = ['#ff5a1f', '#2f6bff', '#12b37a', '#f2c200', '#b44bff', '#00b8cc', '#ff3d7f', '#8cc800', '#ff9d00', '#7a86ff'];

/** Farben der Ebenen innerhalb der gewählten Richtung */
export const PLANE_COLORS = ['#ffffff', '#ffd23f', '#3ee0ff', '#ff6bd6', '#7dff8a', '#c9a7ff', '#ff9f43', '#a3b4ff'];

export function distToSegment(p: Point, a: Point, b: Point): number {
	const dx = b.x - a.x;
	const dy = b.y - a.y;
	const len2 = dx * dx + dy * dy;
	const t = len2 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2)) : 0;
	return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
}

/** Linie um `length` in beide Richtungen verlängern */
export function extend(a: Point, b: Point, length: number): [Point, Point] {
	const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
	const ux = ((b.x - a.x) / len) * length;
	const uy = ((b.y - a.y) / len) * length;
	return [
		{ x: a.x - ux, y: a.y - uy },
		{ x: b.x + ux, y: b.y + uy }
	];
}

export function uniqueId(prefix: string, taken: { id: string }[]): string {
	const ids = new Set(taken.map((t) => t.id));
	let i = taken.length + 1;
	while (ids.has(`${prefix}${i}`)) i++;
	return `${prefix}${i}`;
}

export const pct = (v: number) => `${(v * 100).toFixed(1).replace('.', ',')} %`;
