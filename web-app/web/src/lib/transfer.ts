import { imageSize } from './api';
import type { EbenenGruppe, Line, Point, Richtung, Scene } from './types';

/** Kennung der Exportdatei */
const FORMAT = 'geometrie-analyse';
const VERSION = 1;

/** Exportdatei: Bild als Data-URL plus die Gruppenstruktur (gleiches Format wie die Scene der Auswertung). */
export interface ExportFile extends Scene {
	format: typeof FORMAT;
	version: number;
	name: string;
	imageDataUrl: string;
}

export function exportJson(name: string, src: string, scene: Scene) {
	const data: ExportFile = { format: FORMAT, version: VERSION, name, imageDataUrl: src, ...scene };
	const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
	const a = document.createElement('a');
	a.href = url;
	a.download = `${name || 'analyse'}.json`;
	a.click();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Exportdatei einlesen. Versteht auch das ältere Beispielformat (parallelLineGroups / parallelPlane /
 * perpendicularThree, Linien direkt in den Gruppen). */
export async function importJson(file: Blob, fallbackName: string): Promise<{ name: string; src: string; scene: Scene }> {
	let data: any;
	try {
		data = JSON.parse(await file.text());
	} catch {
		throw new Error('Keine gültige JSON-Datei.');
	}
	const src = data?.imageDataUrl;
	if (typeof src !== 'string' || !src.startsWith('data:image/')) throw new Error('Die Datei enthält kein Bild (imageDataUrl).');

	const scene = Array.isArray(data.parallelLineGroups) ? fromLegacy(data) : fromScene(data);
	const { width, height } = await imageSize(src);
	// Koordinaten an die tatsächliche Bildgröße anpassen, falls die Angabe abweicht
	const sx = scene.imageWidth ? width / scene.imageWidth : 1;
	const sy = scene.imageHeight ? height / scene.imageHeight : 1;
	if (sx !== 1 || sy !== 1)
		for (const l of scene.lines)
			for (const p of [l.point1, l.point2]) [p.x, p.y] = [p.x * sx, p.y * sy];
	[scene.imageWidth, scene.imageHeight] = [width, height];

	return { name: typeof data.name === 'string' && data.name ? data.name : fallbackName, src, scene };
}

const isPoint = (p: any): p is Point => Number.isFinite(p?.x) && Number.isFinite(p?.y);
const point = (p: Point): Point => ({ x: +p.x, y: +p.y });

function fromScene(d: any): Scene {
	if (!Array.isArray(d.lines) || !Array.isArray(d.richtungen)) throw new Error('Unbekanntes Format: lines/richtungen fehlen.');
	const lines: Line[] = d.lines
		.filter((l: any) => typeof l?.id === 'string' && isPoint(l.point1) && isPoint(l.point2))
		.map((l: any) => ({ id: l.id, point1: point(l.point1), point2: point(l.point2) }));
	const lineIds = new Set(lines.map((l) => l.id));
	const richtungen: Richtung[] = d.richtungen.map((r: any) => ({
		id: String(r.id),
		name: String(r.name ?? r.id),
		ebenen: (r.ebenen ?? []).map((e: any) => ({
			id: String(e.id),
			name: String(e.name ?? e.id),
			lineIds: (e.lineIds ?? []).filter((id: string) => lineIds.has(id))
		}))
	}));
	const ebenen = new Set(richtungen.flatMap((r) => r.ebenen.map((e) => e.id)));
	const groups = (gs: any): EbenenGruppe[] =>
		(Array.isArray(gs) ? gs : []).map((g: any) => ({
			id: String(g.id),
			name: String(g.name ?? g.id),
			ebenen: (g.ebenen ?? []).filter((id: string) => ebenen.has(id))
		}));
	return {
		imageWidth: +d.imageWidth || 0,
		imageHeight: +d.imageHeight || 0,
		lines,
		richtungen,
		horizonte: groups(d.horizonte),
		senkrecht: groups(d.senkrecht)
	};
}

/** Altes Format: Richtungen mit Ebenen und eingebetteten Linien, Gruppen über Namen */
function fromLegacy(d: any): Scene {
	const lines: Line[] = [];
	/** gleiche Linie in mehreren Ebenen (Kante) nur einmal anlegen */
	const byCoords = new Map<string, string>();
	const richtungByName = new Map<string, Richtung>();
	const ebeneByName = new Map<string, string>();

	const richtungen: Richtung[] = d.parallelLineGroups.map((g: any, i: number) => {
		const r: Richtung = { id: `g${i + 1}`, name: String(g.id ?? `Richtung ${i + 1}`), ebenen: [] };
		r.ebenen = (g.planeParallelLineGroups ?? []).map((p: any, j: number) => {
			const lineIds: string[] = [];
			for (const l of p.lines ?? []) {
				if (!isPoint(l?.point1) || !isPoint(l?.point2)) continue;
				const key = [l.point1.x, l.point1.y, l.point2.x, l.point2.y].join();
				let id = byCoords.get(key);
				if (!id) {
					id = `n${lines.length + 1}`;
					byCoords.set(key, id);
					lines.push({ id, point1: point(l.point1), point2: point(l.point2) });
				}
				if (!lineIds.includes(id)) lineIds.push(id);
			}
			const e = { id: `${r.id}.${j + 1}`, name: String(p.id ?? `Ebene ${j + 1}`), lineIds };
			if (!ebeneByName.has(e.name)) ebeneByName.set(e.name, e.id);
			return e;
		});
		richtungByName.set(r.name, r);
		return r;
	});

	/** Senkrecht: Name einer Richtung -> alle ihre Ebenen; Ebenen-Gruppe: Name einer Ebene, sonst größte Ebene der Richtung */
	const resolve = (name: string, wholeRichtung: boolean): string[] => {
		const r = richtungByName.get(name);
		const e = ebeneByName.get(name);
		if (wholeRichtung && r) return r.ebenen.map((x) => x.id);
		if (e) return [e];
		if (!r?.ebenen.length) return [];
		return [r.ebenen.reduce((a, b) => (b.lineIds.length > a.lineIds.length ? b : a)).id];
	};
	const groups = (gs: any, prefix: string, item: string, wholeRichtung: boolean): EbenenGruppe[] =>
		(Array.isArray(gs) ? gs : []).map((names: any, k: number) => ({
			id: `${prefix}${k + 1}`,
			name: `${item} ${k + 1}`,
			ebenen: [...new Set((Array.isArray(names) ? names : []).flatMap((n: any) => resolve(String(n), wholeRichtung)))]
		}));

	return {
		imageWidth: +d.imageWidth || 0,
		imageHeight: +d.imageHeight || 0,
		lines,
		richtungen,
		// parallelPlane nennt Ebenen, perpendicularThree Richtungen (alle ihre Ebenen, wie beim Ziehen im Editor)
		horizonte: groups(d.parallelPlane, 'e', 'Ebenen-Gruppe', false),
		senkrecht: groups(d.perpendicularThree, 's', 'Senkrecht', true)
	};
}
