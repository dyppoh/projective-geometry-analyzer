import { base } from '$app/paths';
import type { Detector, Result, Scene } from './types';

async function post<T>(path: string, body: unknown): Promise<T> {
	const res = await fetch(`${base}/api/${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	const data = await res.json().catch(() => null);
	if (!res.ok) throw new Error(data?.message ?? `Fehler ${res.status}`);
	return data;
}

/** Automatisch annotieren: Linien (M-LSD bzw. Hough) erkennen und gruppieren (TLC). */
export const annotate = (image: string, detector: Detector) => post<Scene>('annotate', { image, detector });

/** Linien und Gruppen auswerten: Abstandsmaße + Urteil. */
export const evaluate = (scene: Scene) => post<Result>('evaluate', scene);

export function readAsDataUrl(blob: Blob): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(blob);
	});
}

export function imageSize(src: string): Promise<{ width: number; height: number }> {
	const img = new Image();
	img.src = src;
	return img.decode().then(() => ({ width: img.naturalWidth, height: img.naturalHeight }));
}

export interface MllmStatus {
	status: 'offen' | 'abgegeben';
	/** zählt jede Abgabe (erneut abgeben überschreibt) */
	version: number;
	modell?: string;
	notiz?: string;
	hinweise?: string[];
	abgegeben?: string;
	scene?: Scene;
}

/** Gruppierung durch ein MLLM: Linien erkennen (M-LSD bzw. Hough) und eine Aufgabe unter /group/<token> anlegen. */
export const startMllm = (image: string, detector: Detector) =>
	post<{ token: string; lines: number; shown: number; detector: string }>('mllm', { image, detector });

export async function mllmStatus(token: string): Promise<MllmStatus> {
	const res = await fetch(`${base}/api/mllm/${token}`);
	const data = await res.json().catch(() => null);
	if (!res.ok) throw new Error(data?.message ?? `Fehler ${res.status}`);
	return data;
}
