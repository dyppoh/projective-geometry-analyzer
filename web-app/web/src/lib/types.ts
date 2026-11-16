/** Datenformat zwischen Webserver und Python-Service (siehe README). */

export interface Point {
	x: number;
	y: number;
}

export interface Line {
	id: string;
	point1: Point;
	point2: Point;
}

/** Linien einer Richtung auf einer Ebene(nschar) (Rule 2). Eine Linie darf in mehreren Ebenen stehen (Kante). */
export interface Ebene {
	id: string;
	name: string;
	lineIds: string[];
}

/** 3D-Richtung (Rule 1) mit ihren Ebenen */
export interface Richtung {
	id: string;
	name: string;
	ebenen: Ebene[];
}

/** Ebenen-Gruppe (Rule 3/4: Fluchtpunkte auf einem Horizont) bzw. Senkrecht-Gruppe (Rule 5) - Ebenen-IDs */
export interface EbenenGruppe {
	id: string;
	name: string;
	ebenen: string[];
}

/** Gruppenstruktur, die ausgewertet wird - kommt aus der automatischen Annotation, dem MLLM bzw. dem Editor. */
export interface Scene {
	imageWidth: number;
	imageHeight: number;
	/** alle Linien, auch nicht zugeordnete */
	lines: Line[];
	richtungen: Richtung[];
	horizonte: EbenenGruppe[];
	senkrecht: EbenenGruppe[];
}

export interface Layer {
	label: string;
	lineColor: string;
	pointColor: string;
	lines?: { point1: Point; point2: Point }[];
	points?: Point[];
	/** false = Linien nicht verlängern (Default true) */
	extend?: boolean;
}

/** Prüfgruppe der Auswertung (eine Regel auf einer Gruppe). Status: befund = mindestens ein Kriterium
 * über der Grenze, grundwert = Rule 1 (Maß der Messungenauigkeit, wird nicht bewertet), ungeprueft = kein
 * Grundwert, entfallen = Prüfung nicht möglich (z.B. zu wenige Fluchtpunkte) */
export interface CheckGroup {
	id: string;
	rule: string;
	title: string;
	subject: string;
	status: 'befund' | 'ok' | 'grundwert' | 'ungeprueft' | 'entfallen';
	/** Begründung bei entfallen / ungeprueft / grundwert */
	reason: string | null;
	layers: Layer[];
}

/** median: Median der Abstände über der Grenze */
export interface Kriterium {
	wert: 'median';
	faktor: number;
	/** Grenze = Faktor × Grundwert, Anteil der Bildbreite */
	threshold: number;
	value: number;
	befund: boolean;
	over: { ref: string; value: number }[];
}

export interface Finding {
	/** ID der Prüfgruppe */
	group: string;
	rule: string;
	category: string;
	subject: string;
	title: string;
	nSingles: number;
	/** Kriterien mit Befund */
	points: number;
	kriterien: Kriterium[];
}

/** Verfahren der Linienerkennung */
export type Detector = 'mlsd' | 'hough';

export const DETECTORS: Record<Detector, { name: string; hint: string }> = {
	mlsd: { name: 'M-LSD', hint: 'erkennt gezielter Geometrie' },
	hough: { name: 'Hough', hint: 'erkennt mehr und genauere Linien' }
};

export type Decision = 'echt' | 'unecht' | 'nicht bewertbar';

export interface Verdict {
	decision: Decision;
	reason: string;
	/** Zeilen, "- " = Aufzählung */
	explanation: string;
	befunde: number;
	kriterien: number;
	pruefgruppen: number;
	/** Median der Rule-1-Einzelabstände (Anteil der Bildbreite) */
	grundwert: { value: number | null; n: number; richtungen: number } | null;
	findings: Finding[];
}

export interface Result extends Scene {
	verdict: Verdict;
	checks: CheckGroup[];
}
